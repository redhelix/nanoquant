"""Enables `python -m nanoquant.serve` as an alias for serve/serve.py."""
import runpy, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
runpy.run_path(str(Path(__file__).parent.parent / "serve" / "serve.py"), run_name="__main__")
