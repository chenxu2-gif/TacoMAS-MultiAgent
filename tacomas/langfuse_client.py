from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def get_lf_client():
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    base_url = os.getenv("LANGFUSE_BASE_URL", "").strip() or None

    if not public_key or not secret_key:
        return None

    try:
        from langfuse import Langfuse
    except Exception:
        return None

    try:
        return Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=base_url,
        )
    except Exception:
        return None
