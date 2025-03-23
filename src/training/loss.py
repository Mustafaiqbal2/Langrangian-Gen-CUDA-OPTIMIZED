class CrossEntropyLoss:
    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def forward(self, predictions, targets):
        """
        Compute the cross-entropy loss between predictions and targets.

        Args:
            predictions (torch.Tensor): The predicted probabilities (logits).
            targets (torch.Tensor): The ground truth labels.

        Returns:
            torch.Tensor: The computed loss.
        """
        loss = -torch.sum(targets * torch.log(predictions + 1e-12), dim=-1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def backward(self, predictions, targets):
        """
        Compute the gradient of the loss with respect to the predictions.

        Args:
            predictions (torch.Tensor): The predicted probabilities (logits).
            targets (torch.Tensor): The ground truth labels.

        Returns:
            torch.Tensor: The gradient of the loss.
        """
        return predictions - targets