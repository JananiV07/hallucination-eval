"""Test configuration.

Adds ``src/`` to ``sys.path`` so the test-suite runs even when the package has
not been ``pip install -e .``'d. The ``integration`` marker is declared in
``pyproject.toml``.
"""
import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
