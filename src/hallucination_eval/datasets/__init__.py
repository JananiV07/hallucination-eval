"""Dataset loading for hallucination-eval."""
from __future__ import annotations

from .loader import SUPPORTED_DATASETS, load_samples

__all__ = ["load_samples", "SUPPORTED_DATASETS"]
