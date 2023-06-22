import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1

    def forward(self, logits, targets):
        probs = logits.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()

        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice
