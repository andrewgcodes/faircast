"""Fair Cast: unbiased bf16 export + training on the served copy for LLM RL post-training."""
from .cast import (
    ef_cast_bf16,
    ema_update_,
    export_bf16,
    nearest_cast_bf16,
    served_forward,
    sr_cast_bf16,
    ste_params,
    sync_to_sampler,
)

__version__ = "0.1.0"
__all__ = [
    "sr_cast_bf16",
    "ef_cast_bf16",
    "nearest_cast_bf16",
    "ema_update_",
    "export_bf16",
    "sync_to_sampler",
    "ste_params",
    "served_forward",
]
