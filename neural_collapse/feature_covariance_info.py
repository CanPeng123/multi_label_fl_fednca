import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from data.dataset_info import CIFAR10_LABEL_LIST
from data.dataset_info import VOC2012_LABEL_CLASSES, XRAY14_LABEL_CLASSES
from data.dataset_info import CHESTMNIST_LABEL_LIST, CHESTMNIST_LABEL_LIST_W_BACKGROUND, SKIN_LABEL_LIST, CANCER_LABEL_LIST

def compute_classwise_average(features, one_hot_labels, dataset_name="openi",  multiplicity_1=False, seperate_background_class=False):
    """
    Computes the average feature vector for each class, and prints sample counts.

    Args:
        features (Union[np.ndarray, List[Tensor], List[np.ndarray]]): Shape (N, D), where N is samples, D is feature dim.
        one_hot_labels (Union[np.ndarray, List[Tensor], List[np.ndarray]]): Shape (N, C), where C is number of classes.

    Returns:
        List[np.ndarray]: Class-wise average feature vectors, length C.
    """
    if "medmnist_chest" in dataset_name:
        if not seperate_background_class:
            LABEL_LIST = CHESTMNIST_LABEL_LIST
        else:
            LABEL_LIST = CHESTMNIST_LABEL_LIST_W_BACKGROUND
    elif "medmnist_path" in dataset_name:
        LABEL_LIST = CANCER_LABEL_LIST
    elif "medmnist_derma" in dataset_name:
        LABEL_LIST = SKIN_LABEL_LIST
    elif "cifar10" in dataset_name:
        LABEL_LIST = CIFAR10_LABEL_LIST
    elif "voc2012" in dataset_name:
        LABEL_LIST = VOC2012_LABEL_CLASSES
    elif "xray14" in dataset_name:
        LABEL_LIST = XRAY14_LABEL_CLASSES

    # Convert lists of tensors to NumPy arrays
    features = np.stack([f.cpu().numpy() if isinstance(f, torch.Tensor) else f for f in features])
    one_hot_labels = np.stack([l.cpu().numpy() if isinstance(l, torch.Tensor) else l for l in one_hot_labels])

    num_samples = features.shape[0]  # feature size: 768
    num_classes = one_hot_labels.shape[1]
    classwise_averages = []
    print(f"Total number of samples: {num_samples}")
    print("Number of classes: {0}".format(num_classes))
    print("--------------------------- multiplicity_1: {0}".format(multiplicity_1))

    for class_idx in range(num_classes):
        if multiplicity_1:
            # Mask for samples with exactly one label
            single_label_mask = one_hot_labels.sum(axis=1) == 1
            # Mask for samples where the current class is the only label
            class_mask = (one_hot_labels[:, class_idx] == 1) & single_label_mask
        else:
            class_mask = one_hot_labels[:, class_idx] == 1

        # Apply the mask to get features
        class_features = features[class_mask]


        num_class_samples = class_features.shape[0]
        print(f"Class {class_idx} ({LABEL_LIST[class_idx]}): {num_class_samples} samples")

        if num_class_samples > 0:
            avg_feature = np.mean(class_features, axis=0)
        else:
            avg_feature = np.zeros(features.shape[1])
        classwise_averages.append(avg_feature)
    return classwise_averages

def compute_global_mean_feature(classwise_averages):
    """
    Computes the global mean feature vector from class-wise average features.

    Args:
        classwise_averages (List[np.ndarray]): List of class-wise average feature vectors.

    Returns:
        np.ndarray: Global mean feature vector.
    """
    classwise_averages = np.stack(classwise_averages)  # Shape: (C, D)
    global_mean = np.mean(classwise_averages, axis=0)  # Shape: (D,)
    return global_mean

def compute_l2_norms_and_normalized_directions(classwise_averages, global_mean_feature, return_directions=True):
    """
    Computes the L2 norms ||μ_k - μ_G|| for each class mean relative to the global mean.
    Optionally returns the normalized direction vectors as well.

    Args:
        classwise_averages (List[np.ndarray]): List of class-wise average feature vectors.
        global_mean_feature (np.ndarray): Global mean feature vector.
        return_directions (bool): If True, also returns the L2-normalized direction vectors.

    Returns:
        norms (List[float]): L2 norms for each class mean relative to the global mean.
        directions (List[np.ndarray], optional): Unit vectors in the direction of μ_k - μ_G
    """
    norms = []
    directions = []

    for mu_k in classwise_averages:
        diff = mu_k - global_mean_feature
        norm = np.linalg.norm(diff)

        norms.append(norm)
        if return_directions:
            if norm > 0:
                directions.append(diff / norm)
            else:
                directions.append(np.zeros_like(diff))  # avoid division by zero
    return (norms, directions) if return_directions else norms

def compute_direction_cosine_similarity_matrix(directions, similarity_plot_path=None, compared_similarity_plot_path=None):
    """
    Computes pairwise cosine similarity between class-wise direction vectors
    and optionally plots the similarity matrix.

    Args:
        directions (List[np.ndarray]): List of L2-normalized class direction vectors (shape: C × D).
        plot_path (str or None): If provided, saves a heatmap of the similarity matrix.

    Returns:
        np.ndarray: Cosine similarity matrix of shape (C, C)
    """
    num_classes = len(directions)
    directions = np.stack(directions)  # Shape: (C, D)

    # Compute cosine similarity matrix
    similarity_matrix = directions @ directions.T  # shape: (C, C)

    # Optional plot
    if similarity_plot_path is not None:
        os.makedirs(os.path.dirname(similarity_plot_path), exist_ok=True)
        plt.figure(figsize=(6, 5))
        plt.imshow(similarity_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label='Cosine Similarity')
        plt.title('Pairwise Cosine Similarity of Class Directions')
        plt.xlabel('Class Index')
        plt.ylabel('Class Index')
        plt.tight_layout()
        plt.savefig(similarity_plot_path)
        plt.close()

    ideal_angle = - 1 /(num_classes - 1)
    similarity_matrix_compared_w_ETF = similarity_matrix - ideal_angle

    # Optional plot
    if compared_similarity_plot_path is not None:
        os.makedirs(os.path.dirname(compared_similarity_plot_path), exist_ok=True)
        plt.figure(figsize=(6, 5))
        plt.imshow(similarity_matrix_compared_w_ETF, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label='Cosine Similarity')
        plt.title('Pairwise Cosine Similarity of Class Directions compared with ETF distribution')
        plt.xlabel('Class Index')
        plt.ylabel('Class Index')
        plt.tight_layout()
        plt.savefig(compared_similarity_plot_path)
        plt.close()

    return similarity_matrix, similarity_matrix_compared_w_ETF


# ------------------------------------------------------
# ------ compute different covariance information ------
# ------------------------------------------------------
def compute_within_class_covariance(features, one_hot_labels, classwise_averages, plot_path=None, multiplicity_1 = False):
    """
    Computes the within-class covariance matrix and optionally saves a heatmap.

    Args:
        features (Union[np.ndarray, List[Tensor], List[np.ndarray]]): Shape (N, D)
        one_hot_labels (Union[np.ndarray, List[Tensor], List[np.ndarray]]): Shape (N, C)
        classwise_averages (List[np.ndarray]): Length C, each of shape (D,)
        plot_path (str or None): If not None, saves the covariance heatmap to this path.

    Returns:
        np.ndarray: Within-class covariance matrix of shape (D, D)
    """
    # Convert to numpy arrays
    features = np.stack([f.cpu().numpy() if isinstance(f, torch.Tensor) else f for f in features])
    one_hot_labels = np.stack([l.cpu().numpy() if isinstance(l, torch.Tensor) else l for l in one_hot_labels])

    num_classes = one_hot_labels.shape[1]
    feature_dim = features.shape[1]

    # Initialize covariance matrix
    within_class_cov = np.zeros((feature_dim, feature_dim))
    total_samples = 0

    for class_idx in range(num_classes):
        class_mask = one_hot_labels[:, class_idx] == 1
        class_features = features[class_mask]
        n_c = class_features.shape[0]

        if n_c == 0:
            continue

        centered = class_features - classwise_averages[class_idx]
        cov_c = centered.T @ centered
        within_class_cov += cov_c
        total_samples += n_c

    if total_samples > 0:
        within_class_cov /= total_samples
    else:
        print("Warning: No samples to compute within-class covariance.")

    # Plot if path is provided
    if plot_path is not None:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.figure(figsize=(8, 6))
        plt.imshow(within_class_cov, cmap='viridis')
        plt.colorbar(label='Covariance')
        plt.title('Within-Class Covariance Matrix')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Feature Dimension')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    return within_class_cov

def compute_between_class_covariance(classwise_averages, global_mean_feature, plot_path=None, multiplicity_1 = False):
    """
    Computes the between-class covariance matrix and optionally saves a heatmap.

    Args:
        classwise_averages (List[np.ndarray]): List of class-wise average feature vectors, shape (C, D)
        global_mean_feature (np.ndarray): Global mean feature vector, shape (D,)
        plot_path (str or None): If not None, saves the covariance heatmap to this path.

    Returns:
        np.ndarray: Between-class covariance matrix of shape (D, D)
    """
    # Stack classwise averages into array of shape (C, D)
    classwise_averages = np.stack(classwise_averages)
    num_classes, feature_dim = classwise_averages.shape

    # Initialize between-class covariance matrix
    between_class_cov = np.zeros((feature_dim, feature_dim))

    for class_avg in classwise_averages:
        diff = (class_avg - global_mean_feature).reshape(-1, 1)  # shape (D, 1)
        between_class_cov += diff @ diff.T  # outer product

    # Normalize by number of classes
    between_class_cov /= num_classes

    # Optional: plot and save heatmap
    if plot_path is not None:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.figure(figsize=(8, 6))
        plt.imshow(between_class_cov, cmap='viridis')
        plt.colorbar(label='Covariance')
        plt.title('Between-Class Covariance Matrix')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Feature Dimension')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    return between_class_cov


def compute_total_covariance(features, one_hot_labels, global_mean_feature, plot_path=None, multiplicity_1 = False):
    """
    Computes the total covariance matrix of all samples with respect to the global mean.

    Args:
        features (Union[np.ndarray, List[Tensor], List[np.ndarray]]): Shape (N, D)
        one_hot_labels (Union[np.ndarray, List[Tensor], List[np.ndarray]]): Shape (N, C)
        global_mean_feature (np.ndarray): Global mean feature vector, shape (D,)
        plot_path (str or None): If not None, saves the covariance heatmap to this path.

    Returns:
        np.ndarray: Total covariance matrix of shape (D, D)
    """
    # Convert to numpy arrays
    features = np.stack([f.cpu().numpy() if isinstance(f, torch.Tensor) else f for f in features])
    one_hot_labels = np.stack([l.cpu().numpy() if isinstance(l, torch.Tensor) else l for l in one_hot_labels])

    num_samples, feature_dim = features.shape

    # Subtract global mean from all features
    centered_features = features - global_mean_feature.reshape(1, -1)  # (N, D)

    # Compute total covariance
    total_cov = centered_features.T @ centered_features  # (D, D)
    total_cov /= num_samples  # normalize

    # Optional: plot and save heatmap
    if plot_path is not None:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.figure(figsize=(8, 6))
        plt.imshow(total_cov, cmap='viridis')
        plt.colorbar(label='Covariance')
        plt.title('Total Covariance Matrix')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Feature Dimension')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    return total_cov


# ------------------------------------------------------------------------
# ------ compute class-wise prototype cosine similarity information ------
# ------------------------------------------------------------------------
def compute_distance_alignment_matrix(classwise_averages, global_mean_feature, plot_path=None):
    """
    Computes a matrix of pairwise absolute differences in distances from class means to the global mean.

    Args:
        classwise_averages (List[np.ndarray]): List of class-wise average feature vectors.
        global_mean_feature (np.ndarray): Global mean feature vector.
        plot_path (str or None): If provided, saves a heatmap of the difference matrix.

    Returns:
        np.ndarray: A symmetric matrix (C x C) where entry (i, j) is 
                    | ||μ_i - μ_G|| - ||μ_j - μ_G|| |
    """
    num_classes = len(classwise_averages)
    distances = np.array([
        np.linalg.norm(mu_k - global_mean_feature)
        for mu_k in classwise_averages
    ])

    diff_matrix = np.abs(distances[:, None] - distances[None, :])  # Shape: (C, C)

    # Optional heatmap plot
    if plot_path is not None:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.figure(figsize=(6, 5))
        plt.imshow(diff_matrix, cmap='viridis')
        plt.colorbar(label='| Distance Difference |')
        plt.title('Pairwise Distance Differences to Global Mean')
        plt.xlabel('Class Index')
        plt.ylabel('Class Index')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    return diff_matrix


