"""Shared helpers: seeding and checkpoint I/O."""

import os
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path, model, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {"model_state_dict": model.state_dict()}
    if extra is not None:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(path, model, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt
#（注：内容由AI生成）
