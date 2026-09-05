import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np

from data.image_datasets.data_augmentation import transforms_train_augmented

XRAY_CLASSES = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema",  "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Pleural Thickening", "No Finding"]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(XRAY_CLASSES)}

def multi_hot_to_class_indices(label_tensor, threshold=0.0):
    return (label_tensor > threshold).nonzero(as_tuple=False).flatten().tolist()

class ChestXray14_ImagesDataset(Dataset):
    def __init__(self, split: str, dataset_dir: str, label_dir: str, task_key: str, image_size=(224, 224), img_augmentation=False):
        self.image_size = image_size
        self.img_augmentation = img_augmentation
        self.split = split
        self.num_labels = len(XRAY_CLASSES)

        self.base_dir = os.path.join(dataset_dir, "XRay14")
        
        # Construct path to task-specific CSV (e.g., client-level label file)
        if "train" in split:
            buff = task_key.split("_")
            label_task_key = "train_step_{0}_client_{1}".format(buff[-3], buff[-1])
            label_csv_path = os.path.join(self.base_dir, "PruneCXR", label_dir, f"{label_task_key}.csv")
        elif "val" in split:
            label_csv_path = os.path.join(self.base_dir, "PruneCXR/original", "miccai2023_nih-cxr-lt_labels_val.csv")
        elif "test" in split:
            label_csv_path = os.path.join(self.base_dir, "PruneCXR/original", "miccai2023_nih-cxr-lt_labels_test.csv")
        else:
            raise ValueError("ChestX-ray14 | Something wrong with the split!")


        # Load label CSV into dict
        label_df = pd.read_csv(label_csv_path)
        label_df.set_index("id", inplace=True)
        self.label_dict = label_df.to_dict(orient="index")

        # Image IDs come directly from CSV index
        self.image_ids = list(self.label_dict.keys())

        self.image_dir = os.path.join(self.base_dir, "images")

        self.raw_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, image_id)

        image = Image.open(img_path).convert("RGB")
        if self.img_augmentation:
            image = transforms_train_augmented(image)

        # Build multi-hot label vector
        label_data = self.label_dict[image_id]
        label_vec = torch.tensor([label_data[cls] for cls in XRAY_CLASSES], dtype=torch.float32)
        labels = multi_hot_to_class_indices(label_vec)

        return {
            "images": image,
            "labels": labels,
            "target_scores": label_vec
        }

    def get_label_distribution(self):
        label_counts = np.zeros(len(XRAY_CLASSES), dtype=np.int32)
        for image_id in self.image_ids:
            label_data = self.label_dict[image_id]
            for idx, cls in enumerate(XRAY_CLASSES):
                if label_data[cls] == 1:
                    label_counts[idx] += 1
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

def build_xray14_dataloader(logger, args, split: str, label_dir: str, task_key: str, client_id=-1, **kwargs):
    logger.info(f"ChestX-ray14: Building dataloader for split='{split}', task='{task_key}'")

    shuffle = True if "train" in split.lower() else False
    image_size = getattr(args, "image_size", (224, 224))  # Default to (224, 224) for ChestX-ray14
    augmentation = getattr(args, "img_augmentation", False)

    dataset = ChestXray14_ImagesDataset(
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


