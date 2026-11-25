# Copyright © 2026 Apple Inc.

"""Inkling 2.

https://huggingface.co/thinkingmachines/Inkling-2-Small
"""

from mlx_lm.models.inkling2.config import ModelArgs
from mlx_lm.models.inkling2.model import Model
from mlx_lm.models.inkling2.mtp import (
    AdaptiveMTPController,
    Inkling2DSparkConfig,
    Inkling2DSparkDrafter,
    Inkling2DSparkModel,
)

__all__ = [
    "AdaptiveMTPController",
    "Inkling2DSparkConfig",
    "Inkling2DSparkDrafter",
    "Inkling2DSparkModel",
    "Model",
    "ModelArgs",
]
