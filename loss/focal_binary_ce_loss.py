import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------
# ------ Focal Loss based on Binary Cross Entropy Loss ------
# -----------------------------------------------------------
def calculate_dynamic_alpha_from_dataloader(dataloader):
    # Dynamic alpha calculation based on inverse label frequency
    label_counts = torch.zeros(dataloader.dataset.num_labels)  # Initialize label counts

    # Iterate through the dataset to count occurrences of each label
    for batch in dataloader:
        labels = batch["target_scores"]
        label_counts += labels.sum(dim=0)  # Sum along batch dimension (for each label)
    print("label_counts: {0}".format(label_counts))
    
    total_samples = len(dataloader.dataset)
    print("total_samples: {0}".format(total_samples))
    
    # Calculate inverse frequencies
    alpha = total_samples / (label_counts + 1e-8)  # Avoid division by zero
    # Normalize alpha to ensure it sums to 1
    alpha = alpha / alpha.sum()

    return alpha

class FocalLoss_BinaryCE(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss_BinaryCE, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        print("loss | FocalLoss_BinaryCE | alpha: {0}".format(self.alpha))

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)  # Pt for each class

        loss = self.alpha * (1 - p_t) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

