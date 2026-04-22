"""
W4A16 dequant-GEMV kernel for Nemotron-Nano-9B-v2 decode path.

Optimized for Ampere (RTX 3090, SM86) — also works on Ada / Hopper.

Actual decode shapes for Nemotron-Nano-9B-v2 (confirmed from live profiling):
  in_proj:  x=[1,4480]  W=[22656,4480]  -> y=[1,22656]
  out_proj: x=[1,10240] W=[4480,10240]  -> y=[1,4480]

Weight storage: int4 packed 2-per-byte, COLUMN-MAJOR [K//2, N].
Scales/zeros:  per-group bfloat16, COLUMN-MAJOR [K//gs, N].

Column-major layout coalesces adjacent-thread reads (same K position, adjacent N):
  row-major [N, K//2]: adjacent threads access rows K_half bytes apart → uncoalesced
  col-major [K//2, N]: adjacent threads access adjacent bytes → coalesced → ~1.5x BW

Kernel design — K-parallel 2D grid (pid_n, pid_g):
  Each CTA handles 1 quantization group (GS_HALF=64 inner iterations).
  Groups reduce via atomicAdd into a float32 accumulator, then cast to bf16.
  Exposes n_g * ceil(N/BLOCK_N) parallelism — e.g. 6195 CTAs for in_proj.
"""

import torch
import triton
import triton.language as tl


def quantize_w4(W: torch.Tensor, group_size: int = 128):
    """
    Asymmetric int4 quantization, packed 2-per-byte, column-major storage.

    Args:
        W:          [out_features, in_features]  float or bfloat16
        group_size: K columns per quantization group (must divide in_features)

    Returns:
        W_q:    [in_features//2, out_features]         uint8     col-major
        scales: [in_features//group_size, out_features] bfloat16  col-major
        zeros:  [in_features//group_size, out_features] bfloat16  additive zero
    """
    assert W.dim() == 2, "W must be 2D [out, in]"
    N, K = W.shape
    assert K % group_size == 0, f"in_features ({K}) must be divisible by group_size ({group_size})"

    W = W.float()
    n_g = K // group_size
    W_g = W.reshape(N, n_g, group_size)

    w_min = W_g.amin(dim=-1, keepdim=True)
    w_max = W_g.amax(dim=-1, keepdim=True)
    scale = (w_max - w_min) / 15.0
    scale = scale.clamp(min=1e-8)
    zero  = w_min / scale

    W_int = ((W_g / scale - zero).round().clamp(0, 15)).to(torch.uint8)

    W_int_flat = W_int.reshape(N, K)
    W_lo = W_int_flat[:, 0::2]
    W_hi = W_int_flat[:, 1::2]
    W_packed   = W_lo | (W_hi << 4)
    W_packed_T = W_packed.T.contiguous()        # [K//2, N] col-major

    scale_2d = scale.squeeze(-1)                # [N, n_g]
    zero_2d  = zero.squeeze(-1)                 # [N, n_g]

    return (
        W_packed_T,
        scale_2d.T.contiguous().to(torch.bfloat16),   # [n_g, N]
        zero_2d.T.contiguous().to(torch.bfloat16),    # [n_g, N]
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 32},  num_warps=1, num_stages=1),
        triton.Config({"BLOCK_N": 32},  num_warps=2, num_stages=1),
        triton.Config({"BLOCK_N": 64},  num_warps=2, num_stages=1),
        triton.Config({"BLOCK_N": 64},  num_warps=4, num_stages=1),
        triton.Config({"BLOCK_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_N": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_N": 256}, num_warps=8, num_stages=1),
    ],
    key=["N", "K"],
)
@triton.jit
def _gemv_w4a16_kernel(
    x_ptr,        # [K]       bfloat16
    W_ptr,        # [K//2, N] uint8   col-major
    s_ptr,        # [n_g, N]  bfloat16 col-major
    z_ptr,        # [n_g, N]  bfloat16 col-major
    y_ptr,        # [N]       float32  (atomic accumulator)
    N, K, n_g, K_half,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N:    tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    GS_HALF: tl.constexpr = GROUP_SIZE // 2

    g_N   = pid_g * N
    scale = tl.load(s_ptr + g_N + n_offs, mask=n_mask, other=1.0).to(tl.float32)
    zero  = tl.load(z_ptr + g_N + n_offs, mask=n_mask, other=0.0).to(tl.float32)

    k_base  = pid_g * GROUP_SIZE
    kb_base = (k_base // 2) * N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    for j in range(GS_HALF):
        x_lo_j = tl.load(x_ptr + k_base + j * 2    ).to(tl.float32)
        x_hi_j = tl.load(x_ptr + k_base + j * 2 + 1).to(tl.float32)
        w_packed = tl.load(W_ptr + kb_base + j * N + n_offs, mask=n_mask, other=0).to(tl.uint8)
        w_lo = ((w_packed & 0xF).to(tl.float32) + zero) * scale
        w_hi = ((w_packed >> 4 ).to(tl.float32) + zero) * scale
        acc += w_lo * x_lo_j + w_hi * x_hi_j

    tl.atomic_add(y_ptr + n_offs, acc, mask=n_mask)


def _chunked_linear_impl(
    x_2d:  torch.Tensor,   # [B, K] bfloat16 or float32
    W_q:   torch.Tensor,   # [K//2, N] uint8   col-major
    scales: torch.Tensor,  # [K//gs, N] bfloat16
    zeros:  torch.Tensor,  # [K//gs, N] bfloat16
    group_size: int = 128,
    chunk_groups: int = 8,
) -> torch.Tensor:
    """BF16 dequant matmul for prefill / large-batch decode. Chunked to cap peak VRAM."""
    N   = W_q.shape[1]
    K   = x_2d.shape[1]
    n_g = K // group_size
    x_bf16 = x_2d.to(torch.bfloat16)
    y = torch.zeros(x_2d.shape[0], N, dtype=torch.bfloat16, device=x_2d.device)

    for g_start in range(0, n_g, chunk_groups):
        g_end  = min(g_start + chunk_groups, n_g)
        n_gc   = g_end - g_start
        k_len  = n_gc * group_size
        k_s    = g_start * group_size

        W_chunk = W_q[k_s // 2: (k_s + k_len) // 2]
        w_lo    = (W_chunk & 0xF).float()
        w_hi    = (W_chunk >> 4).float()
        w_int   = torch.stack([w_lo, w_hi], dim=1).reshape(k_len, N)

        sc = scales[g_start:g_end].float().unsqueeze(1).expand(n_gc, group_size, N)
        zr = zeros[g_start:g_end].float().unsqueeze(1).expand(n_gc, group_size, N)
        W_dq = ((w_int.reshape(n_gc, group_size, N) + zr) * sc).reshape(k_len, N).T.contiguous()

        y += torch.nn.functional.linear(x_bf16[:, k_s: k_s + k_len], W_dq.to(torch.bfloat16))
    return y


@torch.library.custom_op("nanoquant::linear", mutates_args=())
def w4a16_linear(
    x:      torch.Tensor,   # [B, K] bfloat16
    W_q:    torch.Tensor,   # [K//2, N] uint8   col-major
    scales: torch.Tensor,   # [K//gs, N] bfloat16
    zeros:  torch.Tensor,   # [K//gs, N] bfloat16
    group_size: int,
) -> torch.Tensor:
    """
    W4A16 GEMV via Triton. Custom op so torch.compile sees it as a leaf —
    body runs eagerly during CUDA graph capture, recording per-batch-size kernels.

    batch=1  → B sequential Triton GEMVs (fast decode, memory-bound)
    batch>1  → same loop; use with --enable-chunked-prefill to bound chunk size
    """
    K      = x.shape[1]
    N      = W_q.shape[1]
    n_g    = K // group_size
    K_half = K // 2
    rows   = []
    for i in range(x.shape[0]):
        y_f32 = torch.zeros(N, dtype=torch.float32, device=x.device)
        grid  = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), n_g)
        _gemv_w4a16_kernel[grid](
            x[i], W_q, scales, zeros, y_f32,
            N, K, n_g, K_half,
            GROUP_SIZE=group_size,
        )
        rows.append(y_f32.to(torch.bfloat16))
    return torch.stack(rows, dim=0)


@w4a16_linear.register_fake
def _w4a16_linear_fake(x, W_q, scales, zeros, group_size):
    return x.new_empty(x.shape[0], W_q.shape[1], dtype=torch.bfloat16)


@torch.library.custom_op("nanoquant::linear_decode", mutates_args=("scratch",))
def w4a16_linear_decode(
    x:      torch.Tensor,   # [1, K] bfloat16  — decode only (batch=1)
    W_q:    torch.Tensor,   # [K//2, N] uint8   col-major
    scales: torch.Tensor,   # [K//gs, N] bfloat16
    zeros:  torch.Tensor,   # [K//gs, N] bfloat16
    group_size: int,
    scratch: torch.Tensor,  # [N] float32  pre-allocated — mutated in place, zero allocation
) -> torch.Tensor:
    """
    Decode-only variant that reuses a pre-allocated float32 scratch buffer.
    Eliminates the torch.zeros(N) allocation per layer per decode step,
    preventing CUDA private pool exhaustion under concurrent load.

    Only valid for batch=1 (single decode token). W4A16Linear.forward routes
    batch>1 to w4a16_linear instead.
    """
    K      = x.shape[1]
    N      = W_q.shape[1]
    n_g    = K // group_size
    K_half = K // 2
    scratch.zero_()
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), n_g)
    _gemv_w4a16_kernel[grid](
        x[0], W_q, scales, zeros, scratch,
        N, K, n_g, K_half,
        GROUP_SIZE=group_size,
    )
    return scratch.to(torch.bfloat16).unsqueeze(0)   # [1, N]


@w4a16_linear_decode.register_fake
def _w4a16_linear_decode_fake(x, W_q, scales, zeros, group_size, scratch):
    return x.new_empty(1, W_q.shape[1], dtype=torch.bfloat16)


def gemv_w4a16(
    x:         torch.Tensor,          # [K] or [1, K] bfloat16 on CUDA
    W_q:       torch.Tensor,          # [K//2, N] uint8 col-major on CUDA
    scales:    torch.Tensor,          # [K//gs, N] bfloat16 on CUDA
    zeros:     torch.Tensor,          # [K//gs, N] bfloat16 on CUDA
    group_size: int = 128,
    out:       torch.Tensor = None,   # optional pre-allocated [N] float32 scratch
) -> torch.Tensor:
    """Direct Triton GEMV. Returns [1, N] bfloat16. For use outside vLLM."""
    if x.dim() == 2:
        assert x.shape[0] == 1
        x = x.squeeze(0)
    K = x.shape[0]
    N = W_q.shape[1]
    n_g    = K // group_size
    K_half = K // 2

    if out is not None:
        y_f32 = out
        y_f32.zero_()
    else:
        y_f32 = torch.zeros(N, dtype=torch.float32, device=x.device)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), n_g)
    _gemv_w4a16_kernel[grid](
        x, W_q, scales, zeros, y_f32,
        N, K, n_g, K_half,
        GROUP_SIZE=group_size,
    )
    return y_f32.to(torch.bfloat16).unsqueeze(0)
