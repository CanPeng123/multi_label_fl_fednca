import torch
import torch.nn as nn


class InstanceNorm1dWrapper(nn.Module):
    def __init__(self, num_features, affine=True):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=affine)

    def forward(self, x):
        # Input shape: [B, F] → [B, F, 1]
        x = x.unsqueeze(-1)
        x = self.norm(x)
        # Output shape: [B, F, 1] → [B, F]
        return x.squeeze(-1)


class GroupNorm1dWrapper(nn.Module):
    def __init__(self, num_channels, num_groups=32, affine=True):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, affine=affine)

    def forward(self, x):
        # Input: [B, F] → [B, F, 1]
        x = x.unsqueeze(-1)
        x = self.norm(x)
        return x.squeeze(-1)


class PerClassMLPs(nn.Module):
    def __init__(self, num_classes, input_dim, hidden_dim, output_dim, norm_type="batch_norm"):
        super(PerClassMLPs, self).__init__()
        self.num_classes = num_classes
        self.norm_type = norm_type

        print("PerClassMLPs | num_classes: {0}, input_dim: {1}, hidden_dim: {2}, output_dim: {3}, norm_type: {4}".
              format(num_classes, input_dim, hidden_dim, output_dim, norm_type))

        # Create a list of MLPs, one per class
        if norm_type == "batch_norm":
            self.mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim),
                    nn.BatchNorm1d(output_dim, affine=False)
                )
                for _ in range(num_classes)
            ])
        elif norm_type == "layer_norm":
            self.mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim),
                    nn.LayerNorm(output_dim, elementwise_affine=False)  # Matches affine=False from BatchNorm
                )
                for _ in range(num_classes)
            ])
        elif norm_type == "instance_norm":
            self.mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    InstanceNorm1dWrapper(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim),
                    InstanceNorm1dWrapper(output_dim, affine=False)
                )
                for _ in range(num_classes)
            ])
        elif norm_type == "group_norm":
            self.mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    GroupNorm1dWrapper(hidden_dim, num_groups=1),  # adjust groups as needed
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim),
                    GroupNorm1dWrapper(output_dim, num_groups=1, affine=False)
                )
                for _ in range(num_classes)
            ])
        else:
            raise ValueError(f"Unsupported normalization type: {norm_type}")

    def forward(self, x, class_idx):
        if isinstance(class_idx, int):
            if (x.size(0) == 1) and (self.norm_type == "batch_norm"):
                was_training = self.mlps[class_idx].training  # Save state
                self.mlps[class_idx].eval()
                out = self.mlps[class_idx](x)
                if was_training:
                    self.mlps[class_idx].train()  # Restore state
                return out
            else:
                return self.mlps[class_idx](x)
        else:
            raise ValueError("PerClassMLPs | Invalid class_idx: expected int")


def main():
    model = PerClassMLPs(num_classes=5, input_dim=10, hidden_dim=20, output_dim=2)
    sample_input = torch.randn(8, 10)
    class_indices = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])
    output = model(sample_input, class_indices)
    print("Output shape:", output.shape)
    print("Output tensor:\n", output)


if __name__ == "__main__":
    main()
