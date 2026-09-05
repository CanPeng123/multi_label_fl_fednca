import torch
from typing import List, Dict
from medmnist import INFO

from modeling.continual_learner import ContinualLearner
from modeling.resnet.resnet_encoder import ResNetEncoderWrapper
from modeling.mlp.mlp import PerClassMLPs
from modeling.attention.attention import ClassAttentionBlock

from neural_collapse.cosine_classifier import FCClassifierLayer
from neural_collapse.etf_classifier import ETFClassifier

class ResNetLearner(ContinualLearner):
    def __init__(self,
                 client_list: List[str],
                 encoder: ResNetEncoderWrapper,
                 encoder_dim: int,
                 task_configs: Dict,
                 device: torch.device,
                 adapter_config: str,
                 dataset_path: str,
                 obtain_class_wise_feature=False,
                 projection_type="class_wise_MLP",
                 seperate_background_class=False,
                 norm="batch_norm",
                 dataset_name='cifar10'
                 ):
        super().__init__()
        self.encoder = encoder
        self.encoder_dim = encoder_dim
        self.device = device
        self.client_list = client_list
        self.task_configs = task_configs
        self.adapter_config = adapter_config
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.obtain_class_wise_feature = obtain_class_wise_feature
        self.projection_type = projection_type
        self.seperate_background_class = seperate_background_class

        buff = client_list[0].split("_")
        task_config_name = "{0}_train".format(buff[0])
        task_config = task_configs[task_config_name]
        self.task_config = task_config
        self.classifier_type = task_config['classifier_type']

        if "medmnist" not in task_config["task_name"]:
            num_labels = task_config["num_labels"]
        else:
            dataset_name = "{0}mnist".format(task_config["images_source"].split("_")[-1])
            num_labels = len(INFO[dataset_name]['label'])
            if self.seperate_background_class and "chest" in dataset_name:
                num_labels = num_labels + 1
                print("resnet_learner | seperate_background_class num_labels + 1")
        self.num_class = num_labels
        num_images = task_config["num_images"]
        print(f"resnet_learner | num_labels: {num_labels}, num_images: {num_images}")

        if task_config["model_type"] != "classification":
            raise ValueError("resnet | Only classification is supported for image-only model")

        # -------------------------------- Classifier --------------------------------
        # ----------------------------------------------------------------------------
        if self.classifier_type == "FC_Classifier":
            self.clf_layer = FCClassifierLayer(self.encoder_dim * num_images, num_images, num_labels, norm_type=norm)
        elif self.classifier_type == "MultiLabel_ETF_Classifier":
            self.clf_layer = ETFClassifier(num_classes=num_labels, feature_dim=self.encoder_dim * num_images, device=self.device)
            self.etf_matrix_value = self.clf_layer.get_etf_matrix()
        elif self.classifier_type == "MultiLabel_ETF_Classifier_w_feature_normalized":
            self.clf_layer = ETFClassifier(num_classes=num_labels, feature_dim=self.encoder_dim * num_images, device=self.device, feature_normalized=True)
            self.etf_matrix_value = self.clf_layer.get_etf_matrix()
        elif self.classifier_type == "Dual_Classifier":
            self.clf_layer = FCClassifierLayer(self.encoder_dim * num_images, num_images, num_labels)
            self.etf_layer = ETFClassifier(num_classes=num_labels, feature_dim=self.encoder_dim * num_images, device=self.device)
            self.etf_layer.eval()
            self.etf_matrix_value = self.etf_layer.get_etf_matrix()
        else:
            raise ValueError("resnet | Invalid classifier type!")
        
        # ---------------------------- Class-specific MLP to obtain class-specific feature ----------------------------
        # -------------------------------------------------------------------------------------------------------------
        if self.obtain_class_wise_feature:
            if self.projection_type == "class_wise_MLP":
                self.class_wise_MLP_layer = PerClassMLPs(num_classes=num_labels, input_dim=self.encoder_dim * num_images, hidden_dim=self.encoder_dim * 2, output_dim=self.encoder_dim * num_images, norm_type=norm)
            elif self.projection_type == "Attention_learnable_random_init":
                self.attention_layer = ClassAttentionBlock(feature_dim=self.encoder_dim, num_classes=num_labels, class_embed_dim=self.encoder_dim, num_heads=4, use_learnable_embeddings=True)
            elif self.projection_type == "Attention_learnable_etf_init":
                print("self.etf_matrix_value size: {0}".format(self.etf_matrix_value.size()))
                self.attention_layer = ClassAttentionBlock(feature_dim=self.encoder_dim, num_classes=num_labels, class_embed_dim=self.encoder_dim, num_heads=4, use_learnable_embeddings=True, 
                                                           predefined_class_embeddings=self.etf_matrix_value)
            elif self.projection_type == "Attention_fixed_etf_init":
                self.attention_layer = ClassAttentionBlock(feature_dim=self.encoder_dim, num_classes=num_labels, class_embed_dim=self.encoder_dim, num_heads=4, use_learnable_embeddings=False, 
                                                           predefined_class_embeddings=self.etf_matrix_value)
            elif self.projection_type == "DETR_Attention_fixed_etf_init":
                self.attention_layer = ClassAttentionBlock(feature_dim=self.encoder_dim, num_classes=num_labels, class_embed_dim=self.encoder_dim, num_heads=4, use_learnable_embeddings=False, 
                                                           predefined_class_embeddings=self.etf_matrix_value)
            else:
                raise ValueError("resnet | Invalid class-wise feature projection type!")


    def forward(self, client_key: str, images: List, label_mask: List = None):
        task_config = self.task_config

        assert task_config["model_type"] == "classification"

        if task_config["num_images"] == 1:
            return self.forward_single_image(images, label_mask=label_mask)
        else:
            return self.forward_multi_images(images, task_config["num_images"], label_mask=label_mask)

    def forward_single_image(self, images: List, capture_activations: bool = False, label_mask: List = None):
        x = self.encoder.process_inputs(images).to(self.device)
        if self.obtain_class_wise_feature and "Attention" in self.projection_type:
            features = self.encoder.forward_get_feature(x)  # (B, C, H, W)
        else:
            features = self.encoder(x)  # [B, D]

        if self.obtain_class_wise_feature and "MLP" in self.projection_type:
            if self.classifier_type == "Dual_Classifier":
                total_cls_logits = []
                total_cls_etf_logits = []
                total_cls_feature = []
                concate_cls_logits = torch.zeros(features.size(0), self.num_class, device=features.device)
                concate_cls_etf_logits = torch.zeros(features.size(0), self.num_class, device=features.device)
                for cls_index in range(self.num_class):
                    cls_features = self.class_wise_MLP_layer(features, cls_index)
                    cls_logits = self.clf_layer(cls_features)
                    cls_etf_logits = self.etf_layer(cls_features)
                    
                    cls_logits_mask = torch.zeros_like(cls_logits)
                    cls_logits_mask[:, cls_index] = cls_logits[:, cls_index]  # zero out all logits except the one for cls_index
                    concate_cls_logits += cls_logits_mask  # accumulate into concate_cls_logits

                    cls_etf_logits_mask = torch.zeros_like(cls_etf_logits)
                    cls_etf_logits_mask[:, cls_index] = cls_etf_logits[:, cls_index]  # zero out all logits except the one for cls_index
                    concate_cls_etf_logits += cls_etf_logits_mask  # accumulate into concate_cls_etf_logits

                    total_cls_logits.append(cls_logits)
                    total_cls_feature.append(cls_features)
                    total_cls_etf_logits.append(cls_etf_logits)
                total_cls_logits_tensor = torch.stack(total_cls_logits)  
                total_cls_etf_logits_tensor = torch.stack(total_cls_etf_logits)  
                total_cls_feature_tensor = torch.stack(total_cls_feature)
                return total_cls_feature_tensor, concate_cls_logits, total_cls_logits_tensor, concate_cls_etf_logits, total_cls_etf_logits_tensor
            else:
                total_cls_logits = []
                total_cls_feature = []
                logits = torch.zeros(features.size(0), self.num_class, device=features.device)
                for cls_index in range(self.num_class):
                    cls_features = self.class_wise_MLP_layer(features, cls_index)
                    cls_logits = self.clf_layer(cls_features)
                    
                    mask = torch.zeros_like(cls_logits)  # zero out all logits except the one for cls_index
                    mask[:, cls_index] = cls_logits[:, cls_index]
                    logits += mask  # accumulate into final_logits

                    total_cls_logits.append(cls_logits)
                    total_cls_feature.append(cls_features)
                total_cls_logits_tensor = torch.stack(total_cls_logits)  
                total_cls_feature_tensor = torch.stack(total_cls_feature)
                return total_cls_feature_tensor, logits, total_cls_logits_tensor
        elif self.obtain_class_wise_feature and "Attention" in self.projection_type:
            if self.classifier_type == "Dual_Classifier":
                total_cls_feature = self.attention_layer(features)  # (B, Channel, H, W) -> (B, num_class, feature_len)
                
                total_cls_logits = []
                total_cls_etf_logits = []
                concate_cls_logits = torch.zeros(features.size(0), self.num_class, device=features.device)
                concate_cls_etf_logits = torch.zeros(features.size(0), self.num_class, device=features.device)

                for cls_index in range(self.num_class):
                    cls_logits = self.clf_layer(total_cls_feature[:, cls_index])
                    cls_etf_logits = self.etf_layer(total_cls_feature[:, cls_index])
                    
                    cls_logits_mask = torch.zeros_like(cls_logits)
                    cls_logits_mask[:, cls_index] = cls_logits[:, cls_index]  # zero out all logits except the one for cls_index
                    concate_cls_logits += cls_logits_mask  # accumulate into concate_cls_logits

                    cls_etf_logits_mask = torch.zeros_like(cls_etf_logits)
                    cls_etf_logits_mask[:, cls_index] = cls_etf_logits[:, cls_index]  # zero out all logits except the one for cls_index
                    concate_cls_etf_logits += cls_etf_logits_mask  # accumulate into concate_cls_etf_logits

                    total_cls_logits.append(cls_logits)
                    total_cls_etf_logits.append(cls_etf_logits)
                total_cls_logits_tensor = torch.stack(total_cls_logits)  
                total_cls_etf_logits_tensor = torch.stack(total_cls_etf_logits)  
                
                return total_cls_feature, concate_cls_logits, total_cls_logits_tensor, concate_cls_etf_logits, total_cls_etf_logits_tensor
            else:
                total_cls_feature, attn_maps, cos_maps = self.attention_layer(features)  # (B, Channel, H, W) -> (B, num_class, feature_len)

                total_cls_logits = []
                logits = torch.zeros(features.size(0), self.num_class, device=features.device)
                
                for cls_index in range(self.num_class):
                    cls_logits = self.clf_layer(total_cls_feature[:, cls_index])
                    mask = torch.zeros_like(cls_logits)  # zero out all logits except the one for cls_index
                    mask[:, cls_index] = cls_logits[:, cls_index]
                    logits += mask  # accumulate into final_logits
                    total_cls_logits.append(cls_logits)
                total_cls_logits_tensor = torch.stack(total_cls_logits)  

                total_cls_feature = total_cls_feature.permute(1, 0, 2)  # (B, num_class, feature_len) -> (num_class, B, feature_len)

                return total_cls_feature, logits, total_cls_logits_tensor, attn_maps, cos_maps
        else:
            logits = self.clf_layer(features)
            if self.classifier_type == "Dual_Classifier":
                etf_logits = self.etf_layer(features)
                return features, logits, etf_logits
            else:
                return features, logits


    def forward_multi_images(self, images: List[List], num_images: int, capture_activations: bool = False, label_mask: List = None):
        batch_size = len(images)
        flat_images = [img for sublist in images for img in sublist]
        x = self.encoder.process_inputs(flat_images).to(self.device)
        x = x.view(batch_size * num_images, 3, x.size(-2), x.size(-1))

        features = self.encoder(x)  # [B*num_images, D]
        features = features.view(batch_size, num_images * self.encoder_dim)
        logits = self.clf_layer(features)
        if self.classifier_type == "Dual_Classifier":
            etf_logits = self.etf_layer(features)
            return features, logits, etf_logits
        else:
            return features, logits

