import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None, reduction='mean'):
        """
        Focal Loss for binary classification.

        Parameters:
        - alpha: Scalar factor to balance the importance of positive/negative examples.
        - gamma: Focusing parameter that reduces the loss for well-classified examples.
        - weight: A manual rescaling weight given to each class. If given, it has to be a Tensor of size `C`.
        - reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.weight)

    def forward(self, inputs, targets):
        # Compute BCE with logits loss
        BCE_loss = self.criterion(inputs, targets)
        # Calculate the probability
        pt = torch.exp(-BCE_loss)
        # Apply the focal loss formula
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss
