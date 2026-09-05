import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def accelerate_and_free_cache(accelerator):
    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()
    accelerator.free_memory()    

def compute_mcc(tp, tn, fp, fn):
    """Calculate Matthews Correlation Coefficient (MCC)."""
    tp = np.float64(tp)
    fp = np.float64(fp)
    tn = np.float64(tn)
    fn = np.float64(fn)
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return 0  # Avoid division by zero
    return numerator / denominator

def compute_acc(tp, tn, fp, fn):
    """Calculate Accuracy (ACC)."""
    total = tp + tn + fp + fn
    if total == 0:
        return 0  # Avoid division by zero
    return (tp + tn) / total

def compute_precision(tp, tn, fp, fn):
    """Calculate Precision."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    return precision

def compute_recall(tp, tn, fp, fn):
    """Calculate Recall."""
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return recall

def compute_specificity(tp, tn, fp, fn):
    """Calculate Specificity."""
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return specificity

def compute_f1(precision, recall):
    """Calculate F1 score."""
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def compute_auc(label_list, soft_pred_list):
    """Calculate AUC."""
    try:
        auc = roc_auc_score(label_list, soft_pred_list)
    except ValueError:
        auc = 0.0  # If only one class is present, set AUC to a default value (e.g., 0.0)
    return auc