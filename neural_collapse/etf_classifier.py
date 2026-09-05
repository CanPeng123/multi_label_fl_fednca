import torch
import torch.nn as nn
import torch.nn.functional as F


class ETFClassifier(nn.Module):
    def __init__(self, num_classes, feature_dim, device, feature_normalized=False, use_bias=True, init_scale=10.0, learnable_scale=True):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.device = device
        self.feature_normalized = feature_normalized

        self.etf_classifier = nn.Linear(feature_dim, num_classes, bias=False).to(device)

        I = torch.eye(num_classes)
        one = torch.ones(num_classes, num_classes)
        weight = torch.sqrt(torch.tensor(num_classes/(num_classes-1))) * (I-(1/num_classes)*one)
        weight /= torch.sqrt((1/num_classes * torch.norm(weight, 'fro')**2))
        weight = torch.mm(weight, torch.eye(num_classes, feature_dim))
        self.etf_classifier.weight = nn.Parameter(weight)
        self.etf_classifier.weight.requires_grad_(False)

        # logit scale
        if learnable_scale:
            self.logit_scale = nn.Parameter(torch.tensor(init_scale))
        else:
            self.register_buffer("logit_scale", torch.tensor(init_scale))

        # per-class bias
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(num_classes))
        else:
            self.register_buffer("bias", torch.zeros(num_classes))

    def get_etf_matrix(self):
        return self.etf_classifier.weight

    def forward(self, x):
        if self.feature_normalized:
            x = F.normalize(x, p=2, dim=1)
            w = F.normalize(self.etf_classifier.weight, p=2, dim=1)
            logits = x @ w.t()
        else:
            logits = self.etf_classifier(x)

        # scale + bias
        logits = self.logit_scale * logits + self.bias
        return logits
