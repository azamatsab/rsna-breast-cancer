from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def partial_mixup(
    input: torch.Tensor, gamma: float, indices: torch.Tensor
) -> torch.Tensor:
    if input.size(0) != indices.size(0):
        raise RuntimeError("Size mismatch!")
    perm_input = input[indices]
    return input.mul(gamma).add(perm_input, alpha=1 - gamma)

def mixup(
    input: torch.Tensor,
    target: torch.Tensor,
    gamma: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randperm(input.size(0), device=input.device, dtype=torch.long)
    return partial_mixup(input, gamma, indices), partial_mixup(target, gamma, indices)

def shuffle_minibatch(x, y):
    assert x.size(0) == y.size(0)
    indices = torch.randperm(x.size(0))
    return x[indices], y[indices]

def cutmix(x_train, y_train, size):
    width, height = size
    x_train_shuffled, y_train_shuffled = shuffle_minibatch(x_train, y_train)
    lam = np.random.beta(1.0, 1.0)
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(width * cut_rat)
    cut_h = np.int(height * cut_rat)

    cx = np.random.randint(width)
    cy = np.random.randint(height)

    bbx1 = np.clip(cx - cut_w // 2, 0, width)
    bby1 = np.clip(cy - cut_h // 2, 0, height)
    bbx2 = np.clip(cx + cut_w // 2, 0, width)
    bby2 = np.clip(cy + cut_h // 2, 0, height)
    
    x_train[:, :, bbx1:bbx2, bby1:bby2] = x_train_shuffled[:, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - (bbx2 - bbx1) * (bby2 - bby1) / (width * height)
    return x_train, lam, y_train_shuffled

def cutmix_criterion(y_preds, y_train, lam, y_train_shuffled, criterion):
    loss = criterion(y_preds, y_train) * lam + criterion(y_preds, y_train_shuffled) * (1. - lam)
    return loss
