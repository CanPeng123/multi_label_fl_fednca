import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from typing import List


class ResNetEncoderWrapper(nn.Module):
    def __init__(self,
                 backbone: nn.Module,
                 encoder_dim: int,
                 device: torch.device,
                 imagenet_pretrain: bool):
        """
        Wrapper for torchvision ResNet-based image encoders.
        This is the image-only counterpart to ViLTEncoderWrapper.

        Args:
            backbone: CNN feature extractor (without classifier head).
            encoder_dim: output feature dimension (e.g., 2048 for ResNet-50).
            device: torch.device.
        """
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.encoder_dim = encoder_dim
        self.device = device
        self.to(device)

        if imagenet_pretrain:
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet-pretrained models (e.g., ResNet18), the official ImageNet means and stds 
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),  # covert the image dimension from HWC format to CHW format
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Training from scratch or small-scale datasets
            ])

    def process_inputs(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Preprocess a list of PIL images into a batch tensor.

        Args:
            images: List of PIL images.

        Returns:
            torch.Tensor of shape (B, 3, H, W).
        """

        tensor_batch = [self.image_transform(img) for img in images]
        return torch.stack(tensor_batch).to(self.device)

    def forward(self, images: torch.Tensor) -> torch.FloatTensor:
        """
        Args:
            images: torch.Tensor of shape (B, 3, H, W)

        Returns:
            torch.FloatTensor: shape (B, encoder_dim)
        """
        with torch.set_grad_enabled(self.training):
            features = self.backbone(images)           # (B, C, H, W)
            pooled = self.pool(features)               # (B, C, 1, 1)
            flat = pooled.squeeze(-1).squeeze(-1)      # (B, C)
            return flat
        
    def forward_get_feature(self, images: torch.Tensor) -> torch.FloatTensor:
        """
        Args:
            images: torch.Tensor of shape (B, 3, H, W)

        Returns:
            torch.FloatTensor: shape (B, encoder_dim)
        """
        with torch.set_grad_enabled(self.training):
            features = self.backbone(images)           # (B, C, H, W)
            return features

