from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42) -> None:
    """모든 랜덤 시드를 고정하여 재현성 보장.

    Args:
        seed: 랜덤 시드 값.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    manual_seed_fn = getattr(torch, "manual_seed", None)
    if callable(manual_seed_fn):
        _ = manual_seed_fn(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"모든 시드를 {seed}로 고정했습니다.")
