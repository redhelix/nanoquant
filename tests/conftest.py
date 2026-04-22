import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")


@pytest.fixture(scope="session")
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    return torch.device("cuda")
