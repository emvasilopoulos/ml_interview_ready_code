import torch


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 1.5):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.bce_loss = torch.nn.BCELoss()

    def forward(self, inputs, targets):
        p = inputs
        ce_loss = self.bce_loss(inputs, targets)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean(1).sum()
