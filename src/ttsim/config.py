from __future__ import annotations

try:
    import jax
except ImportError:
    IS_JAX_INSTALLED = False
else:
    IS_JAX_INSTALLED = True
