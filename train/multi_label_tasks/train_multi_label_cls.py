import torch
from torch import nn
import argparse
import sys
from typing import Dict
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
sys.path.insert(0, ".")

from data.image_datasets.cifar10_dataset import build_cifar10_dataloader
from data.image_datasets.med_mnist_dataset import build_medmnist_dataloader
from data.image_datasets.voc_dataset import build_voc2012_dataloader
from data.image_datasets.xray_dataset import build_xray14_dataloader
from data.image_datasets.coco_dataset import build_coco2017_dataloader

from train.multi_label_tasks.task_trainer import TaskTrainer
from utils.seed_utils import set_seed

from loss.focal_binary_ce_loss import FocalLoss_BinaryCE, calculate_dynamic_alpha_from_dataloader
from loss.binary_ce_w_reject_n_feature_contrastive_loss import BinaryCE_wRejectContrastiveLoss, compute_pos_weight_from_loader


class MultiLabelTrainer(TaskTrainer):
    def __init__(self, logger, args: argparse.Namespace, task_configs: Dict, model_config: Dict, device: torch.device, task_key: str, task_output_dir=None, client_id=-1, accelerator=None):
        """
        Initializes a Trainer that handles training of a model on the VQA task

        args: Arguments provided by user
        task_configs: dictionary containing task-specific configuration parameters for all tasks
        model_config: dictionary containing model-specific configuration parameters
        device: cuda/cpu
        """
        super().__init__()
        self.accelerator = accelerator
        self.device = self.accelerator.device
        set_seed(args.seed + self.accelerator.process_index)   # make sure different process gets different seed
        self.logger = logger

        if args.do_wandb_logging:  # Create W&B experiment
            if self.accelerator.is_main_process:
                self.accelerator.init_trackers(project_name="missing_modality")
                self.accelerator.trackers[0].run.name = (task_output_dir.split("/")[-4] + "/" + task_output_dir.split("/")[-3] + "/" + task_output_dir.split("/")[-1])

        self.args = args
        self.local_epochs = args.local_epochs
        self.task_key = task_key
        self.task_output_dir = task_output_dir
        self.task_config = task_configs[args.task_config_key]
        self.batch2inputs_converter = model_config["batch2inputs_converter"]
        self.classifier_type = args.classifier_type
        self.obtain_class_wise_feature = args.obtain_class_wise_feature
        self.projection_type = args.projection_type

        buff = task_key.split('_')
        self.client_key = '{0}_client_{1}'.format(buff[0], buff[-1])

        # ------------ Create dataloaders for training, validation and test ------------
        # # ------------------------------------------------------------------------------
        self.model_name = args.encoder_name
        self.visual_input_type = model_config["visual_input_type"]  # pil_image
        
        if 'medmnist' in task_key:
            self.train_dataloader, train_dataset = build_medmnist_dataloader(
                logger=self.logger, args=args, split=self.args.splits[0], task_key=self.task_key, client_id=client_id)
            self.val_dataloader, val_dataset = build_medmnist_dataloader(
                logger=self.logger, args=args, split=self.args.splits[1], task_key=self.task_key, client_id=client_id)
            self.test_dataloader, test_dataset = build_medmnist_dataloader(
                logger=self.logger, args=args, split=self.args.splits[2], task_key=self.task_key, client_id=client_id)
        elif 'cifar10' in task_key:
            self.train_dataloader, train_dataset = build_cifar10_dataloader(
                logger=self.logger, args=args, split=self.args.splits[0], label_dir=args.json_text_folder, 
                task_key=self.task_key, client_id=client_id)
            self.val_dataloader, val_dataset = build_cifar10_dataloader(
                logger=self.logger, args=args, split=self.args.splits[1], label_dir=args.json_text_folder, 
                task_key=self.task_key, client_id=client_id)
            self.test_dataloader, test_dataset = build_cifar10_dataloader(
                logger=self.logger, args=args, split=self.args.splits[2], label_dir=args.json_text_folder,
                task_key=self.task_key, client_id=client_id)
        elif 'voc2012' in task_key:
            self.train_dataloader, train_dataset = build_voc2012_dataloader(
                logger=self.logger, args=args, split=self.args.splits[0], label_dir=args.json_text_folder, 
                task_key=self.task_key, client_id=client_id)
            self.val_dataloader, val_dataset = build_voc2012_dataloader(
                logger=self.logger, args=args, split=self.args.splits[1], label_dir=args.json_text_folder, 
                task_key=self.task_key, client_id=client_id)
            self.test_dataloader, test_dataset = build_voc2012_dataloader(
                logger=self.logger, args=args, split=self.args.splits[2], label_dir=args.json_text_folder,
                task_key=self.task_key, client_id=client_id)
        elif 'xray14' in task_key:
            self.train_dataloader, train_dataset = build_xray14_dataloader(
                logger=self.logger, args=args, split=self.args.splits[0], label_dir=args.json_text_folder, 
                task_key=self.task_key, client_id=client_id)
            self.val_dataloader, val_dataset = build_xray14_dataloader(
                logger=self.logger, args=args, split=self.args.splits[1], label_dir=args.json_text_folder, 
                task_key=self.task_key, client_id=client_id)
            self.test_dataloader, test_dataset = build_xray14_dataloader(
                logger=self.logger, args=args, split=self.args.splits[2], label_dir=args.json_text_folder,
                task_key=self.task_key, client_id=client_id)
        elif 'coco' in task_key:
            self.train_dataloader, train_dataset = build_coco2017_dataloader(
                logger=self.logger, args=args, split=self.args.splits[0], label_dir=args.json_text_folder, 
                task_key=self.task_key, client_id=client_id)
            self.val_dataloader, val_dataset = build_coco2017_dataloader(
                logger=self.logger, args=args, split=self.args.splits[1], label_dir=args.json_text_folder, 
                task_key=self.task_key, client_id=client_id)
            self.test_dataloader, test_dataset = build_coco2017_dataloader(
                logger=self.logger, args=args, split=self.args.splits[2], label_dir=args.json_text_folder,
                task_key=self.task_key, client_id=client_id)
        else:
            raise ValueError("train_multi_label_cls | Invalid dataset!")
        
        self.num_of_training_data = len(train_dataset)
        self.num_of_validation_data = len(val_dataset)
        self.num_of_testing_data = len(test_dataset)
        self.label_distribution_training = train_dataset.get_label_distribution()
        self.label_distribution_validation = val_dataset.get_label_distribution()
        self.label_distribution_testing = test_dataset.get_label_distribution()

        logger.info("Dataset | {}: len={}, {}: len={}, {}: len={}".format(
            self.args.splits[0], self.num_of_training_data, self.args.splits[1], self.num_of_validation_data, self.args.splits[2], self.num_of_testing_data))
        # ------------------------------------------------------------------------------

        # -------------------------- Training hyperparameters --------------------------
        # ------------------------------------------------------------------------------
        self.lr = self.args.lr
        self.adam_epsilon = self.task_config["adam_epsilon"]
        self.weight_decay = self.task_config["weight_decay"]
        self.loss_type = args.loss_type
        self.reg_warm_up = args.reg_warm_up

        pos_weight = None
        if args.bce_w_pos_weight:
            if hasattr(args, "global_pos_weight") and args.global_pos_weight is not None:
                pos_weight = args.global_pos_weight.to(self.device)
                print(f"[pos_weight] using GLOBAL pos_weight: min={pos_weight.min().item():.3f}, max={pos_weight.max().item():.3f}")
            else:
                num_classes = self.task_config["num_labels"]
                pos_weight = compute_pos_weight_from_loader(self.train_dataloader, num_classes=num_classes, device=self.device, max_clip=20.0)
                print(f"[pos_weight] using CLIENT pos_weight: min={pos_weight.min().item():.3f}, max={pos_weight.max().item():.3f}")

        if args.loss_type == "binary_ce":
            self.loss_criterion = nn.BCEWithLogitsLoss(reduction='none')  # binary cross-entropy with sigmoid built in
        elif args.loss_type == "binary_ce_w_prob":
            self.loss_criterion = torch.nn.BCELoss(reduction='none')
        elif args.loss_type == "focal_binary_ce":
            class_wise_alpha = calculate_dynamic_alpha_from_dataloader(self.train_dataloader).to(self.device)
            self.loss_criterion = FocalLoss_BinaryCE(alpha=class_wise_alpha, reduction='none')  # focal binary cross-entropy
        elif args.loss_type == "focal_binary_ce_no_alpha":
            class_wise_alpha = 1
            self.loss_criterion = FocalLoss_BinaryCE(alpha=class_wise_alpha, reduction='none')  # focal binary cross-entropy
        elif args.loss_type == "binary_ce_w_dual_reg_rejection_n_contrastive_all_PSC":
            self.loss_criterion = BinaryCE_wRejectContrastiveLoss(reduction='none', rejection_type="all", contrastive_type="PSC", 
                                                                  rejection_margin=self.args.rejection_loss_threshold, 
                                                                  hyparam_rejection=self.args.hyparam_rejection_loss, 
                                                                  hyparam_contractive=self.args.hyparam_contractive_loss,
                                                                  use_focal_bce=args.focal_bce_flag)
        elif args.loss_type == "binary_ce_w_dual_reg_rejection_n_contrastive_all_cosine":
            self.loss_criterion = BinaryCE_wRejectContrastiveLoss(reduction='none', rejection_type="all", contrastive_type="Cosine", 
                                                                  rejection_margin=self.args.rejection_loss_threshold, 
                                                                  hyparam_rejection=self.args.hyparam_rejection_loss, 
                                                                  hyparam_contractive=self.args.hyparam_contractive_loss,
                                                                  use_focal_bce=args.focal_bce_flag)
        elif args.loss_type == "binary_ce_w_dual_reg_rejection_n_contrastive_topK_PSC":
            self.loss_criterion = BinaryCE_wRejectContrastiveLoss(reduction='none', rejection_type="topK", contrastive_type="PSC", 
                                                                  rejection_margin=self.args.rejection_loss_threshold, 
                                                                  hyparam_rejection=self.args.hyparam_rejection_loss, 
                                                                  hyparam_contractive=self.args.hyparam_contractive_loss,
                                                                  use_focal_bce=args.focal_bce_flag, 
                                                                  use_pos_weight=args.bce_w_pos_weight, pos_weight=pos_weight, 
                                                                  reg_warm_up=args.reg_warm_up, total_round=args.comm_rounds)
        elif args.loss_type == "binary_ce_w_dual_reg_rejection_n_contrastive_topK_cosine":
            self.loss_criterion = BinaryCE_wRejectContrastiveLoss(reduction='none', rejection_type="topK", contrastive_type="Cosine", 
                                                                  rejection_margin=self.args.rejection_loss_threshold, 
                                                                  hyparam_rejection=self.args.hyparam_rejection_loss, 
                                                                  hyparam_contractive=self.args.hyparam_contractive_loss,
                                                                  use_focal_bce=args.focal_bce_flag)
        elif (args.loss_type == "cross_entropy") and (not self.obtain_class_wise_feature):
            self.loss_criterion = nn.CrossEntropyLoss(reduction='none')  # cross-entropy with softmax built in
        elif  args.loss_type == "mse_loss":
            self.loss_criterion = nn.MSELoss(reduction='none')
        else:
            raise ValueError("train_multi_label_cls | Invalid loss type!")
        print("train_multi_label_cls | MultiLabelTrainer | loss_type: {0}".format(args.loss_type))

        if self.classifier_type == "Dual_Classifier":
            if self.obtain_class_wise_feature:
                self.etf_reg_loss = nn.BCEWithLogitsLoss(reduction='none')  # binary cross-entropy with sigmoid built in 
            else:
                self.etf_reg_loss = nn.CrossEntropyLoss(reduction='none')  # cross-entropy with softmax built in

        self.num_epochs = self.args.num_epochs

    def train_step(self, model, step, batch, optimizer=None, scheduler=None, hooks=None, epoch=None):
        loss_dict = {}

        if isinstance(batch, dict) and "target_scores" in batch.keys():
            target = batch["target_scores"].to(self.device)
        
            if self.obtain_class_wise_feature:
                if self.classifier_type == "Dual_Classifier":
                    total_cls_feature, logits, total_cls_logits, etf_logits, total_cls_etf_logits = self.forward_pass(model, batch, do_eval=False)
                elif "Attention" in self.projection_type:
                    total_cls_feature, logits, total_cls_logits, attn_maps, cos_maps = self.forward_pass(model, batch, do_eval=False)
                else:
                    total_cls_feature, logits, total_cls_logits = self.forward_pass(model, batch, do_eval=False)
            else:
                if self.classifier_type == "Dual_Classifier":
                    features, logits, etf_logits = self.forward_pass(model, batch, do_eval=False)
                else:
                    # features: encoder feature; logits: sigmoid predtion probability
                    features, logits = self.forward_pass(model, batch, do_eval=False)

            label_counts = target.sum(dim=1)
            
            if self.obtain_class_wise_feature:
                _base_model = model.module if hasattr(model, "module") else model
                if "binary_ce_w_dual_reg_rejection_n_contrastive" in self.loss_type:
                    prototypes = _base_model.clf_layer.get_etf_matrix()
                    if self.reg_warm_up:
                        loss_ce = self.loss_criterion(logits, total_cls_logits, total_cls_feature, target, prototypes,
                        current_round=epoch)
                    else:
                        loss_ce = self.loss_criterion(logits, total_cls_logits, total_cls_feature, target, prototypes)
                elif "binary_ce_w_feature_contrastive" in self.loss_type:
                    prototypes = _base_model.clf_layer.get_etf_matrix()
                    loss_ce = self.loss_criterion(logits, total_cls_logits, total_cls_feature, target, prototypes)
                elif ("binary_ce_w_rejection" in self.loss_type) or self.loss_type == "dual_binary_ce":
                    loss_ce = self.loss_criterion(logits, total_cls_logits, target)
                elif "binary_ce" in self.loss_type or "mse" in self.loss_type:
                    loss_ce = self.loss_criterion(logits, target)
                else: # cross_entropy or angular penalty
                    loss_ce = self.loss_criterion(total_cls_logits, target)
            else:
                loss_ce = self.loss_criterion(logits, target)

            if isinstance(loss_ce, torch.Tensor) and loss_ce.dim() == 2:
                loss_ce = loss_ce.mean(dim=0).sum()                 # scalar

            else:
                # If loss_ce is already [B] (e.g., your BinaryCE_wRejectContrastiveLoss),
                # or scalar, do your existing reduction logic safely:
                if loss_ce.dim() == 1:
                    loss_ce = loss_ce.mean()
                else:
                    # scalar already
                    loss_ce = loss_ce
            
            # -------------------------- regularization --------------------------
            # --------------------------------------------------------------------
            peak_reg = 0
            peak_reg_hparam = 500

            if self.classifier_type == "Dual_Classifier":
                etf_reg = self.etf_reg_loss(etf_logits, target)
                if self.obtain_class_wise_feature:
                    etf_reg = etf_reg.mean(dim=0).sum()  # mean over batch, sum over classes
                else:
                    etf_reg = etf_reg.mean(dim=0)  # mean over batch
            else:
                etf_reg = 0
            etf_reg_hparam = 1

            center_loss = 0
            center_loss_hparam = 0.01

            hnm_loss = 0
            hnm_loss_hparam = 0.1

            loss = loss_ce + (peak_reg_hparam * peak_reg)  + (etf_reg_hparam * etf_reg) + (center_loss_hparam * center_loss) + (hnm_loss_hparam * hnm_loss)
            self.accelerator.backward(loss)

            if optimizer is not None:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

            loss_dict["loss_ce"] = loss_ce
            loss_dict["etf_reg"] = etf_reg
            loss_dict["center_loss_reg"] = center_loss
            loss_dict["hnm_loss_reg"] = hnm_loss
            loss_dict["loss_total"] = loss
            return loss_dict

    def compute_f1_score_with_logits(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        probabilities = torch.sigmoid(logits)  # Apply sigmoid to logits to get probabilities
        preds = (probabilities > 0.5).float()  # Binarize the probabilities with a threshold of 0.5
        preds_np = preds.cpu().numpy()  # Convert tensors to numpy arrays for scikit-learn's f1_score function
        labels_np = labels.cpu().numpy()
        
        # zero_division: Sets the value to return when there is a zero division
        f1 = f1_score(labels_np, preds_np, average='macro', zero_division=0)  # Calculate F1 score
        try:
            auc = roc_auc_score(labels_np, preds_np, average='macro', multi_class='ovr')
        except ValueError:
            auc = 0.0  # If only one class is present, set AUC to a default value (e.g., 0.0)
        return f1, auc

    def get_prediction_probability(self, logits: torch.Tensor, labels: torch.Tensor, return_raw_pred=False) -> float:
        if self.args.get_pred_probability_function == 'sigmoid':
            probabilities = torch.sigmoid(logits)  # Apply sigmoid to logits to get probabilities
        elif self.args.get_pred_probability_function == 'softmax':
            probabilities = F.softmax(logits, dim=1)  # Apply softmax to logits to get probabilities
        elif self.args.get_pred_probability_function == 'non':
            probabilities = logits
            max_val_out = torch.max(probabilities)  # Find max and min
            min_val_out = torch.min(probabilities)
            # print("probabilities | max_val_out: {0}, min_val_out: {1}".format(max_val_out, min_val_out))
        else:
            raise ValueError("train_multi_label_cls MultiLabelTrainer | Invalid get_pred_probability_function: {0}!".format(self.args.get_pred_probability_function))

        labels_np = labels.cpu().numpy()
        soft_preds_np = probabilities.cpu().numpy()
        num_classes = labels_np.shape[1]  # Number of classes
        label_list = []
        soft_pred_list = []
        for i in range(num_classes):  # Prediction and label list for each class 
            soft_pred_list.append(soft_preds_np[:, i])
            label_list.append(labels_np[:, i])
        if not return_raw_pred:
            return label_list, soft_pred_list
        else:
            return label_list, soft_pred_list, probabilities

    def get_num_of_training_data(self):
        return self.num_of_training_data

    def get_training_data_label_distribution(self):
        return self.label_distribution_training
