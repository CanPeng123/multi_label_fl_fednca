import torch
from typing import List, Dict
from torchvision import models
import torch.nn as nn

from modeling.resnet.resnet_encoder import ResNetEncoderWrapper
from modeling.resnet.resnet_learner import ResNetLearner


def get_norm_layer(norm_type="layer_norm"):
    if "group" in norm_type:
        return lambda num_channels: nn.GroupNorm(32, num_channels)
    elif "layer" in norm_type:
        class LayerNorm2d(nn.Module):
            def __init__(self, num_channels, eps=1e-5):
                super().__init__()
                self.norm = nn.LayerNorm(num_channels, eps=eps)

            def forward(self, x):
                # x: [B, C, H, W] → [B, H, W, C]
                x = x.permute(0, 2, 3, 1)
                x = self.norm(x)
                # back to [B, C, H, W]
                return x.permute(0, 3, 1, 2)

        return lambda num_channels: LayerNorm2d(num_channels)

    else:
        raise ValueError(f"Unsupported norm_type: {norm_type}")

def load_resnet_encoder(logger, checkpoint_name: str, device: torch.device, imagenet_pretrain: bool, backbone: str = 'resnet50', norm: str = "batch_norm"):
    """
    Loads a ResNet encoder with optional checkpoint override.

    Args:
        logger: logging instance
        checkpoint_name: path to a checkpoint or "imagenet" for default pretrained weights
        device: torch.device
        backbone: ResNet variant: resnet18, resnet34, resnet50, etc.

    Returns:
        encoder: ResNetEncoderWrapper instance
    """
    logger.info("-" * 100)
    logger.info(f"resnet | load_resnet_encoder | Loading ResNet encoder: {backbone}")

    print("-------------------- load_resnet_encoder | norm: {0}".format(norm))

    if norm != "batch_norm":
        norm_layer = get_norm_layer(norm)

    # Load ResNet backbone choose from: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
    resnet_fn = getattr(models, backbone)
    if norm == "batch_norm":
        model = resnet_fn(pretrained=(checkpoint_name == "imagenet"))
    else:
        model = resnet_fn(pretrained=(checkpoint_name == "imagenet"), norm_layer=norm_layer)

    # Remove classification head
    modules = list(model.children())[:-2]
    feature_extractor = nn.Sequential(*modules)

    encoder = ResNetEncoderWrapper(feature_extractor, model.fc.in_features, device, imagenet_pretrain)

    if checkpoint_name not in ["imagenet", ""]:
        logger.info(f"resnet | load_resnet_encoder | Loading weights from checkpoint: {checkpoint_name}")
        ckpt = torch.load(checkpoint_name, map_location=device)
        encoder.load_state_dict(ckpt)
        
    logger.info("resnet | load_resnet_encoder | Successfully loaded ResNet encoder")
    return encoder


def create_resnet_model(logger, model_name_or_path: str, client_specific_head: bool, client_list: List[str], model_config: Dict, task_configs: Dict,
                        device: torch.device, dataset_path: str, obtain_class_wise_feature: False, projection_type: str,
                        imagenet_pretrain: False, seperate_background_class: False, backbone_name="resnet18", norm_type="batch_norm", dataset_name='cifar10'):
    
    encoder = load_resnet_encoder(
        logger=logger,
        checkpoint_name=model_name_or_path,
        device=device,
        imagenet_pretrain=imagenet_pretrain,
        backbone=model_config.get("backbone", backbone_name),
        norm=norm_type
    )

    model = ResNetLearner(
            client_list=client_list,
            encoder=encoder,
            encoder_dim=model_config["encoder_dim"],
            task_configs=task_configs,
            device=device,
            adapter_config=model_config['adapter_config'] if 'adapter_config' in model_config else None,
            dataset_path=dataset_path,
            obtain_class_wise_feature=obtain_class_wise_feature,
            projection_type=projection_type,
            seperate_background_class=seperate_background_class,
            norm=norm_type,
            dataset_name=dataset_name
        )

    logger.info("resnet | Successfully created and initialized ResNet Learner model")
    return model


def convert_batch_to_resnet_input_dict(batch: Dict):
    """
    Convert inputs from batch_collate into format consumable by ResNet (image-only).
    """
    return {"images": batch["images"]}


def create_resnet18_model(logger, model_name_or_path: str, client_specific_head: bool, client_list: List[str], model_config: Dict, task_configs: Dict,
                        device: torch.device, dataset_path: str, imagenet_pretrain: False, obtain_class_wise_feature: False, projection_type: str,
                        seperate_background_class: False, norm_type: str, dataset_name: str):
    model = create_resnet_model(logger, model_name_or_path, client_specific_head, client_list, model_config, task_configs, device, dataset_path,
                                obtain_class_wise_feature, projection_type, imagenet_pretrain, seperate_background_class, backbone_name="resnet18", norm_type=norm_type,
                                dataset_name=dataset_name)
    return model


def create_resnet50_model(logger, model_name_or_path: str, client_specific_head: bool, client_list: List[str], model_config: Dict, task_configs: Dict,
                        device: torch.device, dataset_path: str, imagenet_pretrain: False, obtain_class_wise_feature: False, projection_type: str,
                        seperate_background_class: False, norm_type: str, dataset_name: str):
    model = create_resnet_model(logger, model_name_or_path, client_specific_head, client_list, model_config, task_configs, device, dataset_path,
                                obtain_class_wise_feature, projection_type, imagenet_pretrain, seperate_background_class, backbone_name="resnet50", norm_type=norm_type,
                                dataset_name=dataset_name)
    return model