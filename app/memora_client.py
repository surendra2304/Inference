"""
Universal Memora Client for Inference Multi-Model Deliberation Gateway
"""
import os
import sys
from pathlib import Path

try:
    MEMORA_ROOT = Path("d:/FRIDAY Universe/Memora")
    if str(MEMORA_ROOT) not in sys.path:
        sys.path.insert(0, str(MEMORA_ROOT))
    from sdk.memora_client import MemoraClient, memora_client  # type: ignore[import-not-found]
except Exception:
    from .memora_cloud_fallback import MemoraClient, memora_client

__all__ = ["MemoraClient", "memora_client"]
