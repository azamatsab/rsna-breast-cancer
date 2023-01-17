import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()  # mean()? Scared to break other math.
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=2, gamma=3, logits=False):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(
                inputs, targets
            )
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        return torch.mean(F_loss)


class SignedLoss(nn.Module):
    def __init__(self, reduction, threshold=0.5):
        super(SignedLoss, self).__init__()
        self.reduction = reduction
        self.base = nn.BCEWithLogitsLoss(reduction="none")
        self.threshold = threshold

    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        diff[diff >= self.threshold] = 1
        diff[diff < self.threshold] = 0
        diff = diff.bool()
        loss = target - 2 * target * pred + torch.pow(pred, 2)
        loss = torch.pow(loss, (~diff).int())
        loss = loss * self.base(pred, target)
        if self.reduction != "none":
            return torch.mean(pred)
        return loss


def contrast_depth_conv(input, device):
    """compute contrast depth in both of (out, label)"""
    """
        input  32x32
        output 8x32x32
    """

    kernel_filter_list = [
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
        [[0, 1, 0], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 1], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 0], [1, -1, 0], [0, 0, 0]],
        [[0, 0, 0], [0, -1, 1], [0, 0, 0]],
        [[0, 0, 0], [0, -1, 0], [1, 0, 0]],
        [[0, 0, 0], [0, -1, 0], [0, 1, 0]],
        [[0, 0, 0], [0, -1, 0], [0, 0, 1]],
    ]

    kernel_filter = np.array(kernel_filter_list, np.float32)

    kernel_filter = torch.from_numpy(kernel_filter.astype(np.float)).float().to(device)
    # weights (in_channel, out_channel, kernel, kernel)
    kernel_filter = kernel_filter.unsqueeze(dim=1)

    input = input.expand(input.shape[0], 8, input.shape[2], input.shape[3])

    contrast_depth = F.conv2d(input, weight=kernel_filter, groups=8)  # depthwise conv

    return contrast_depth


class Contrast_depth_loss(
    nn.Module
):  # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self, device):
        super(Contrast_depth_loss, self).__init__()
        self.device = device
        return

    def forward(self, out, label):
        """
        compute contrast depth in both of (out, label),
        then get the loss of them
        tf.atrous_convd match tf-versions: 1.4
        """
        contrast_out = contrast_depth_conv(out, self.device)
        contrast_label = contrast_depth_conv(label, self.device)

        criterion_MSE = nn.MSELoss().to(self.device)

        loss = criterion_MSE(contrast_out, contrast_label)
        # loss = torch.pow(contrast_out - contrast_label, 2)
        # loss = torch.mean(loss)

        return loss


def mask_loss(pred_mask, gt_mask):
    pred = F.log_softmax(pred_mask, dim=1)
    loss = F.nll_loss(pred, gt_mask.squeeze(1).long())

    return loss
