import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import torch
import numpy as np

from data.image_datasets.data_augmentation import transforms_train_augmented


def multi_hot_to_class_indices(label_tensor, threshold=0.0):
    """
    Converts a multi-hot (or one-hot) label tensor to a list of class indices.

    Args:
        label_tensor (torch.Tensor): shape (num_classes,), values like 0.0, 1.0, 0.5, etc.
        threshold (float): minimum value to consider a class "active" (default 0.0)

    Returns:
        List[int]: indices of active classes
    """
    return (label_tensor > threshold).nonzero(as_tuple=False).flatten().tolist()


class CIFAR10_ImagesDataset(Dataset):
    def __init__(self, split: str, dataset_dir: str, label_dir: str, task_key: str, image_size=(56, 56), img_augmentation=False):
        """
        Args:
            data_dir (str): Directory containing 'images/' and 'labels.json'.
            image_size (tuple): Desired image size (H, W) for training.
            img_augmentation (bool): Whether to apply augmentations.
        """
        if "train" in split:
            data_dir = "{0}/CIFAR10/cifar10_augmented_trainset".format(dataset_dir)
            self.image_dir = os.path.join(data_dir, "images")
            buff = task_key.split("_")
            label_task_key = "train_step_{0}_client_{1}".format(buff[-3], buff[-1])
            self.label_path = os.path.join(data_dir, "{0}/{1}.json".format(label_dir, label_task_key))

        else:
            data_dir = "{0}/CIFAR10/cifar10_augmented_testset".format(dataset_dir)
            self.image_dir = os.path.join(data_dir, "images")
            self.label_path = os.path.join(data_dir, "labels.json")
        
        self.image_size = image_size
        self.img_augmentation = img_augmentation
        self.num_labels = 10

        with open(self.label_path, "r") as f:
            self.label_dict = json.load(f)

        self.filenames = sorted(self.label_dict.keys())

        self.raw_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.expand(3, -1, -1)),  # Convert grayscale to 3-channel
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)     # Normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(self.image_dir, filename)

        image = Image.open(image_path).convert('RGB')
        if self.img_augmentation:
            image = transforms_train_augmented(image)

        one_hot_label = torch.tensor(self.label_dict[filename], dtype=torch.float32)
        label = multi_hot_to_class_indices(one_hot_label)

        return {
            "images": image, 
            "labels": label,
            "target_scores": one_hot_label,
        }

    def get_label_distribution(self):
        num_labels = len(next(iter(self.label_dict.values())))
        label_counts = np.zeros(num_labels)

        for label_list in self.label_dict.values():
            label_counts += np.array(label_list)
        return label_counts
    

def resnet_batch_collate(batch):
    images = [item["images"] for item in batch]  
    labels = [item["labels"] for item in batch]
    target_scores = [item["target_scores"] for item in batch]  # List[Tensor]

    return {
        "images": images,
        "labels": labels,
        "target_scores": torch.stack(tuple(target_scores))
    }


def build_cifar10_dataloader(logger, args, split: str, label_dir: str, task_key: str, client_id=-1, **kwargs):
    
    logger.info(f"CIFAR10: Building dataloader for split='{split}', task='{task_key}'")

    shuffle = True if "train" in split else False
    image_size = getattr(args, "image_size", (64, 64))  # If args has an attribute image_size, use it. If not, fall back to (64, 64).
    augmentation = getattr(args, "img_augmentation", False)

    dataset = CIFAR10_ImagesDataset(
        split=split,
        dataset_dir=args.data_dir,
        label_dir=label_dir,
        task_key=task_key,
        image_size=image_size,
        img_augmentation=augmentation
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

