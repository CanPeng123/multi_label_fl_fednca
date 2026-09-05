from .resnet.resnet import create_resnet18_model, create_resnet50_model
from .vit.vit import create_vit_t_16_model, create_vit_b_16_model


create_model_map = {
    'resnet18': create_resnet18_model,
    'resnet50': create_resnet50_model,
    'vit-t-16': create_vit_t_16_model,
    'vit-b-16': create_vit_b_16_model
}
