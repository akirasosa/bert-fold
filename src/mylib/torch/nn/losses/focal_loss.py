import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma=2, reduction: str = 'mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6
        self.reduction = reduction

    def forward(self, inputs, target):
        p = torch.sigmoid(inputs)
        p = torch.clamp(p, self.smooth, 1.0 - self.smooth)

        alpha = (1 - self.alpha) + target * (2 * self.alpha - 1)
        focal_weight = alpha * torch.where(target.bool(), 1. - p, p)

        bce = -(target * torch.log(p) + (1.0 - target) * torch.log(1.0 - p))
        loss = bce * focal_weight

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss
        # return F.binary_cross_entropy(p, target, reduction='mean', weight=focal_weight)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = torch.exp(-bce_loss)
        alpha = (1 - self.alpha) + targets * (2 * self.alpha - 1)
        loss = alpha * (1 - p_t) ** self.gamma * bce_loss
        return loss.mean()


# %%
if __name__ == '__main__':
    # %%
    y_hat = torch.zeros(9)
    y_true = torch.cat((torch.ones(5), torch.zeros(4)))

    BinaryFocalLoss(alpha=0.25)(y_hat, y_true), FocalLoss()(y_hat, y_true)

    # %%
    alpha = 0.25
    gamma = 2
    p = torch.sigmoid(y_hat)
    p = torch.clamp(p, 1e-6, 1.0 - 1e-6)
    alpha_factor = (1 - alpha) + y_true * (2 * alpha - 1)
    focal_weight = alpha_factor * torch.where(y_true.bool(), 1. - p, p)
    # focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

    # bce = -(y_true * torch.log(p) + (1.0 - y_true) * torch.log(1.0 - p))
    bce = F.binary_cross_entropy(p, y_true, reduction='mean', weight=focal_weight)
    # bce2 = F.binary_cross_entropy_with_logits(y_hat, y_true, reduction='none')
    print(bce)
    # print(bce2)
    # cls_loss = focal_weight * bce
    # cls_loss.mean()

    # %%
    losses = BinaryFocalLoss(reduction='none')(y_hat, y_true)
    scatter_mean(losses, y_true.long()).mean()

    # %%
    F.binary_cross_entropy(torch.sigmoid(y_hat), y_true), F.binary_cross_entropy_with_logits(y_hat, y_true)
