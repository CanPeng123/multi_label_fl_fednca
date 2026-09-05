import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from PIL import Image

from medmnist import INFO


def multi_hot_to_class_indices(label_tensor, threshold=0.0):
    return (label_tensor > threshold).nonzero(as_tuple=False).flatten().tolist()

def class_indices_to_multi_hot(indices, num_classes):
    """
    Converts a list of class indices into a multi-hot one-hot encoded label tensor.

    Args:
        indices (List[int]): List of class indices.
        num_classes (int): Total number of classes.

    Returns:
        torch.FloatTensor: A multi-hot encoded tensor of shape (num_classes,).
    """
    label = torch.zeros(num_classes, dtype=torch.float32)
    for idx in indices:
        idx = int(idx)  # Ensure index is an integer
        if 0 <= idx < num_classes:
            label[idx] = 1.0
    return label


class MedMNIST_ImagesDataset(Dataset):
    def __init__(self, dataset_name: str, split: str, data_dir: str, task_key: str, image_size=(56, 56), img_augmentation=False, seperate_background_class=False):
        """
        Args:
            dataset_name: name like 'chestmnist'
            split: 'train', 'val', or 'test'
            data_dir: root path to the .npy files
        """
        self.dataset_name = dataset_name
        self.split = split
        self.image_size = image_size
        self.img_augmentation = img_augmentation
        self.seperate_background_class = seperate_background_class
        self.task_type = data_dir.split("/")[-1]
        print("MedMNIST_ImagesDataset | self.task_type : {0}".format(self.task_type))
        print("MedMNIST_ImagesDataset | self.img_augmentation : {0}".format(self.img_augmentation))

        self.info = INFO[dataset_name]
        self.num_labels = len(self.info['label'])
        self.task = self.info['task']
        self.n_channels = self.info['n_channels']
        self.n_samples = self.info['n_samples']
        print("MedMNIST_ImagesDataset | dataset: {0} | task: {1}, num_labels: {2}".format(dataset_name, self.task, self.num_labels))
        print("MedMNIST_ImagesDataset | dataset: {0} | INFO | n_channels: {1}, n_samples: {2}".format(dataset_name, self.n_channels, self.n_samples))
        if self.seperate_background_class and ('chest' in dataset_name):
            self.num_labels = self.num_labels + 1
            print("MedMNIST_ImagesDataset | seperate_background_class | dataset: {0} | task: {1}, num_labels: {2}".format(dataset_name, self.task, self.num_labels))


        # === Load images and labels ===
        if "train" in split:
            buff = task_key.split("_")
            current_client_task_key = "train_step_{0}_client_{1}".format(buff[-3], buff[-1])
            print("MedMNIST_ImagesDataset | current_client_task_key: {0}".format(current_client_task_key))
            image_path = os.path.join(data_dir, "{0}_images.npy".format(current_client_task_key))
            label_path = os.path.join(data_dir, "{0}_labels.npy".format(current_client_task_key))
        elif "val" in split:
            image_path = os.path.join(data_dir, "val_images.npy")
            label_path = os.path.join(data_dir, "val_labels.npy")
        else:
            image_path = os.path.join(data_dir, "test_images.npy")
            label_path = os.path.join(data_dir, "test_labels.npy")

        self.images = np.load(image_path)  # shape: (N, H, W, C) or (N, 28, 28)
        self.labels = np.load(label_path)  # shape: (N, num_labels)
        
        print("MedMNIST_ImagesDataset | dataset: {0} | used number of samples: {1}".format(dataset_name, len(self.images)))

        # Optional augmentation (random flip) applied before base transform
        self.augmentation_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),])
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  # shape: (H, W, C) or (H, W)
        label = self.labels[idx]

        if self.n_channels == 1:
            img = img.squeeze()  # ensure (H, W)
            if img.ndim != 2:
                raise ValueError(f"Expected grayscale image of shape (H, W), got {img.shape}")
            img = np.stack([img] * 3, axis=-1)  # (H, W) → (H, W, 3)
        else:
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(f"Expected RGB image of shape (H, W, 3), got {img.shape}")
        img = Image.fromarray(img)
        img = img.convert('RGB')

        if self.img_augmentation:
            img = self.augmentation_transform(img)

        if self.dataset_name == 'chestmnist' or 'multi_label' in self.task_type:
            one_hot_label = torch.tensor(label, dtype=torch.float32)
            class_indices = multi_hot_to_class_indices(one_hot_label)
        elif (self.dataset_name == 'pathmnist') or (self.dataset_name == 'dermamnist'):
            class_indices = torch.tensor(label, dtype=torch.float32)
            one_hot_label = class_indices_to_multi_hot(class_indices, self.num_labels)

        return {
            "images": img,
            "labels": class_indices,
            "target_scores": one_hot_label,
        }
    
    def get_label_distribution(self):
        # Assuming self.labels has shape (N, num_labels)
        label_counts = np.sum(self.labels, axis=0)
        return label_counts


def resnet_batch_collate(batch):
    images = [item["images"] for item in batch]
    labels = [item["labels"] for item in batch]
    target_scores = [item["target_scores"] for item in batch]

    return {
        "images": images,
        "labels": labels,
        "target_scores": torch.stack(target_scores)
    }

def build_medmnist_dataloader(logger, args, split: str, task_key: str, client_id=-1, **kwargs):
    dataset_name = "{0}mnist".format(args.dataset_name.split("_")[-1])
    
    logger.info(f"MedMNIST: Loading from npy for '{dataset_name}' | split='{split}', task='{task_key}'")

    shuffle = True if "train" in split else False
    image_size = getattr(args, "image_size", (56, 56))
    augmentation = getattr(args, "img_augmentation", False)
    seperate_background_class = getattr(args, "seperate_background_class", False)

    if "chest" in dataset_name:
        dataset_dir = "{0}/MedMNIST-128/{1}_128/{2}".format(args.data_dir, dataset_name, args.json_text_folder)
    else:
        dataset_dir = "{0}/MedMNIST-28/{1}_28/{2}".format(args.data_dir, dataset_name, args.json_text_folder)
    print("build_medmnist_dataloader | dataset_dir: {0}".format(dataset_dir))

    dataset = MedMNIST_ImagesDataset(
        dataset_name=dataset_name,
        split=split,
        data_dir=dataset_dir,
        task_key=task_key,
        image_size=image_size,
        img_augmentation=augmentation,
        seperate_background_class=seperate_background_class
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=resnet_batch_collate,
        pin_memory=True
    )

    return dataloader, dataset
