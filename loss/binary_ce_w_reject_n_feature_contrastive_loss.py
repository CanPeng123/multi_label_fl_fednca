import torch
from torch import nn
import torch.nn.functional as F

from loss.focal_binary_ce_loss import FocalLoss_BinaryCE


def compute_psc_loss(selected_features, class_indices, prototypes, tau=0.07):
    """
    Compute PSC loss with correct denominator (includes all classes).
    Args:
        selected_features: [N, D]
        class_indices: [N]
        prototypes: [C, D]
    Returns:
        psc_loss: [N] per-sample PSC loss
    """
    # Normalize
    selected_features = F.normalize(selected_features, dim=1)
    prototypes = F.normalize(prototypes, dim=1)

    # Similarity matrix [N, C]
    sim_matrix = torch.matmul(selected_features, prototypes.T)

    # Compute logits (softmax inputs)
    logits = sim_matrix / tau  # [N, C]

    # Use cross-entropy style loss: softmax over all classes
    psc_loss = F.cross_entropy(logits, class_indices, reduction='none')  # [N]
    return psc_loss


def compute_pos_weight_from_loader(train_loader, num_classes, device, max_clip=20.0, eps=1e-6):
    pos = torch.zeros(num_classes, dtype=torch.float64)
    total = 0

    for batch in train_loader:
        # adjust this line to your batch format:
        y = batch["target_scores"]  # shape [B, C], 0/1
        y = y.to(torch.float64)
        pos += y.sum(dim=0)
        total += y.size(0)

    neg = total - pos
    pos_weight = (neg + eps) / (pos + eps)  # Nneg/Npos

    # clip to avoid extreme tail domination
    pos_weight = torch.clamp(pos_weight, 1.0, max_clip).to(torch.float32).to(device)
    return pos_weight


class BinaryCE_wRejectContrastiveLoss(nn.Module):
    @staticmethod
    def ramp(p, start, end):
        if p <= start: return 0.0
        if p >= end: return 1.0
        return (p - start) / (end - start)

    def __init__(self, reduction='none', rejection_margin=0.3, hyparam_rejection=1, rejection_type="all", hyparam_contractive=1, contrastive_type="PSC", 
    use_focal_bce=False, use_pos_weight=False, pos_weight=None, reg_warm_up=None, total_round=None):
        super(BinaryCE_wRejectContrastiveLoss, self).__init__()
        """
        Args:
            rejection_type: all, random, topK
            contrastive_type: L2, Cosine, PSC
        """

        self.rejection_margin = rejection_margin
        self.hyparam_rejection = hyparam_rejection
        self.hyparam_contractive = hyparam_contractive
        
        self.contrastive_type = contrastive_type
        self.rejection_type = rejection_type

        if use_focal_bce:
            class_wise_alpha = 1
            self.bce_loss_criterion = FocalLoss_BinaryCE(alpha=class_wise_alpha, reduction='none')  # focal binary cross-entropy
        elif use_pos_weight:
            print("use_pos_weight | pos_weight: {0}".format(pos_weight))
            self.bce_loss_criterion = nn.BCEWithLogitsLoss(reduction=reduction, pos_weight=pos_weight)
        else:
            self.bce_loss_criterion = nn.BCEWithLogitsLoss(reduction=reduction)

        self.reg_warm_up = reg_warm_up
        if self.reg_warm_up:
            self.total_round = total_round
            print("reg_warm_up | total_round: {0}".format(total_round))

    def forward(self, logits, total_cls_logits, total_cls_feature, labels, prototypes, current_round=None):
        # --------------------- BCE loss ---------------------
        # ----------------------------------------------------
        bce_loss = self.bce_loss_criterion(logits, labels)
        bce_loss_per_sample = bce_loss.sum(dim=1)
        
        # ------------- Negative Sample Rejection Regularization & Feature Contractive Regularization -------------
        # ---------------------------------------------------------------------------------------------------------
        # --- Step 1: Extract relevant logits and Features ---
        nonzero_indices = labels.nonzero(as_tuple=False)  # [N_active, 2]

        if nonzero_indices.numel() == 0:
            # only BCE, no rejection/contractive (or set them to zeros)
            return bce_loss_per_sample

        # If a sample has multiple 1s in its label, labels.nonzero() will return one row per active class, even if they come from the same sample.
        batch_indices = nonzero_indices[:, 0]  # [N_active]
        class_indices = nonzero_indices[:, 1]  # [N_active]
       
        # total_cls_logits: [C, B, logit_len] → permute to [B, C, logit_len] for indexing
        total_cls_logits_transposed = total_cls_logits.permute(1, 0, 2)  # [B, C, logit_len]
        # total_cls_feature: [C, B, feature_len] → permute to [B, C, feature_len] for indexing
        total_cls_features_transposed = total_cls_feature.permute(1, 0, 2)  # [B, C, feature_len]
        
        # Selected logits for each (sample, class) pair
        selected_logits = total_cls_logits_transposed[batch_indices, class_indices]  # shape: [N_active, logit_len]
        selected_features = total_cls_features_transposed[batch_indices, class_indices]  # shape: [N_active, feature_len]

        # --- Step 2: Extract leftover logits and Features ---
        B, C, _ = total_cls_logits_transposed.shape  # [B, C, logit_len]
        total_pairs = torch.cartesian_prod(torch.arange(B, device=total_cls_logits.device), torch.arange(C, device=total_cls_logits.device))  # [B*C, 2]
        selected_pairs = torch.stack([batch_indices, class_indices], dim=1)  # [N_active, 2]

        # Use set difference to get leftover indices
        selected_set = set(map(tuple, selected_pairs.tolist()))
        leftover_pairs = [pair for pair in total_pairs.tolist() if tuple(pair) not in selected_set]
        leftover_pairs = torch.tensor(leftover_pairs, dtype=torch.long, device=total_cls_logits.device)  # [N_left, 2]
        left_batch_indices = leftover_pairs[:, 0]
        left_class_indices = leftover_pairs[:, 1]
        
        leftover_logits = total_cls_logits_transposed[left_batch_indices, left_class_indices]  # [N_left, logit_len]
        leftover_features = total_cls_features_transposed[left_batch_indices, left_class_indices]  # [N_left, logit_len]

        number_of_selected_feature = selected_logits.size()[0]
        number_of_leftover_feature = leftover_logits.size()[0]

        # --- Step 3: Compute rejection loss (reduce the effect of negative (background) features) ---
        if self.rejection_type == "random":
            # Randomly choose the same number of leftover logits as selected_logits
            if number_of_leftover_feature >= number_of_selected_feature:
                rand_indices = torch.randperm(number_of_leftover_feature, device=total_cls_logits.device)[:number_of_selected_feature]
                leftover_logits = leftover_logits[rand_indices]
                left_batch_indices = left_batch_indices[rand_indices]
            else:
                # Not enough leftover samples — keep all
                print("Warning: Not enough leftover logits for balanced_random. Using all leftovers.")
        elif self.rejection_type.lower() == "topk":
            # Use top-K highest confidence logits (by max-similarity score) from leftover_logits
            if number_of_leftover_feature >= number_of_selected_feature:
                max_sim = leftover_logits.max(dim=1)[0]  # [N_left]
                topk_values, topk_indices = torch.topk(max_sim, number_of_selected_feature)
                leftover_logits = leftover_logits[topk_indices]
                left_batch_indices = left_batch_indices[topk_indices]
            else:
                print("Warning: Not enough leftover logits for balanced_topK. Using all leftovers.")
        else:
            # Use all logits from leftover_logits
            leftover_logits = leftover_logits
            left_batch_indices = left_batch_indices

        if leftover_logits.numel() > 0:
            max_sim = leftover_logits.max(dim=1)[0]
            max_sim_prob = torch.sigmoid(max_sim)  # convert to probability [0, 1]
            rejection_loss = torch.clamp(max_sim_prob - self.rejection_margin, min=0)
            rejection_loss_per_sample = torch.zeros(labels.size(0), device=total_cls_logits.device)  # [batch_size]
            rejection_loss_per_sample = rejection_loss_per_sample.index_add(0, left_batch_indices, rejection_loss)  # sum per sample
        else:
            rejection_loss_per_sample = torch.tensor(0.0, device=total_cls_logits.device)


        # --- Step 4: Compute contractive loss (pull features closer to its prototype and away from other prototypes) ---
        if self.contrastive_type == "L2":
            # prototypes: [C, feature_dim] — one vector per class

            # Gather the relevant prototypes for each selected feature's class
            selected_prototypes = prototypes[class_indices]  # [N_active, feature_dim]

            # Compute L2 distance between features and their corresponding class prototype
            contractive_distances = F.mse_loss(selected_features, selected_prototypes, reduction='none')  # [N_active, feature_dim]
            contractive_loss_per_pair = contractive_distances.sum(dim=1)  # [N_active]

            # Aggregate loss per sample (if multiple active labels per sample)
            contractive_loss_per_sample = torch.zeros(logits.size(0), device=logits.device)  # [B]
            contractive_loss_per_sample = contractive_loss_per_sample.index_add(0, batch_indices, contractive_loss_per_pair)

        elif self.contrastive_type == "Cosine":
            # prototypes: [C, feature_dim] — one vector per class

            # Gather the relevant prototypes for each selected feature's class
            selected_prototypes = prototypes[class_indices]  # [N_active, feature_dim]

            # Compute cosine similarity between features and their corresponding class prototype
            cosine_sim = F.cosine_similarity(selected_features, selected_prototypes, dim=1)  # [N_active]
            contractive_loss_per_pair = 1 - cosine_sim  # higher when features and prototype are dissimilar
            
            # Aggregate loss per sample (if multiple active labels per sample)
            contractive_loss_per_sample = torch.zeros(logits.size(0), device=logits.device)  # [B]
            contractive_loss_per_sample = contractive_loss_per_sample.index_add(0, batch_indices, contractive_loss_per_pair)

        elif self.contrastive_type == "PSC":
            # Prototype Similarity Contrastive (PSC) Loss
            psc_loss = compute_psc_loss(selected_features=selected_features, class_indices=class_indices, prototypes=prototypes)

            # Aggregate loss per sample (if multiple active labels per sample)
            contractive_loss_per_sample = torch.zeros(logits.size(0), device=logits.device)  # [B]
            contractive_loss_per_sample = contractive_loss_per_sample.index_add(0, batch_indices, psc_loss)

        # ------------------ Combine losses ------------------ 
        # ----------------------------------------------------

        # in each round/step:
        if self.reg_warm_up and (current_round is not None) and (self.total_round is not None):
            p = current_round / max(1, self.total_round - 1)
            con_factor = self.ramp(p, 0.2, 0.6)   # 20%->60%
            rej_factor = self.ramp(p, 0.6, 0.9)   # 60%->90%

            lambda_con = con_factor * self.hyparam_contractive
            lambda_rej = rej_factor * self.hyparam_rejection
        else:
            lambda_con = self.hyparam_contractive
            lambda_rej = self.hyparam_rejection

        total_loss = bce_loss_per_sample + lambda_rej * rejection_loss_per_sample + lambda_con * contractive_loss_per_sample

        return total_loss

        