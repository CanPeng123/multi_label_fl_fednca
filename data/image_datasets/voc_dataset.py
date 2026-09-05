import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np

from data.image_datasets.data_augmentation import transforms_train_augmented

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", 
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]

CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(VOC_CLASSES)}

def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    labels = np.zeros(len(VOC_CLASSES), dtype=np.float32)

    for obj in root.findall("object"):
        cls_name = obj.find("name").text.lower().strip()
        if cls_name in CLASS_TO_IDX:
            labels[CLASS_TO_IDX[cls_name]] = 1.0
    return torch.tensor(labels, dtype=torch.float32)

def multi_hot_to_class_indices(label_tensor, threshold=0.0):
    return (label_tensor > threshold).nonzero(as_tuple=False).flatten().tolist()

class VOC2012_ImagesDataset(Dataset):
    def __init__(self, split: str, dataset_dir: str, label_dir: str, task_key: str, image_size=(224, 224), img_augmentation=False):
        self.image_size = image_size
        self.img_augmentation = img_augmentation
        self.split = split
        self.num_labels = 20

        self.base_dir = os.path.join(dataset_dir, "VOC2012")
        if "train" in split:
            buff = task_key.split("_")
            label_task_key = "train_step_{0}_client_{1}".format(buff[-3], buff[-1])
            data_set_file = os.path.join(self.base_dir, "ImageSets", "{0}".format(label_dir), "{0}.txt".format(label_task_key))
        else:
            data_set_file = os.path.join(self.base_dir, "ImageSets", "{0}".format(label_dir), "val.txt")
       
        self.image_dir = os.path.join(self.base_dir, "JPEGImages")
        self.ann_dir = os.path.join(self.base_dir, "Annotations")

        with open(data_set_file, "r") as f:
            self.image_ids = [x.strip() for x in f.readlines()]

        self.raw_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        ann_path = os.path.join(self.ann_dir, f"{image_id}.xml")

        image = Image.open(img_path).convert("RGB")
        if self.img_augmentation:
            image = transforms_train_augmented(image)

        target_scores = parse_voc_xml(ann_path)
        labels = multi_hot_to_class_indices(target_scores)

        return {
            "images": image,
            "labels": labels,
            "target_scores": target_scores
        }
    
    def get_label_distribution(self):
        """
        Computes the distribution of classes in the dataset split.
        
        Returns:
            np.ndarray: An array of shape (num_classes,) with counts of each class.
        """
        label_counts = np.zeros(len(VOC_CLASSES), dtype=np.int32)

        imageset_dir = os.path.join(self.base_dir, "ImageSets", "Main")
        print("get_label_distribution | imageset_dir: {0}".format(imageset_dir))

        for class_name, class_idx in CLASS_TO_IDX.items():
            if "train" in self.split:
                class_file = os.path.join(imageset_dir, f"{class_name}_{self.split}.txt")
            else:
                class_file = os.path.join(imageset_dir, f"{class_name}_val.txt")
            if not os.path.isfile(class_file):
                continue

            with open(class_file, "r") as f:
                for line in f:
                    img_id, label = line.strip().split()
                    if img_id in self.image_ids and int(label) == 1:
                        label_counts[class_idx] += 1

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

def build_voc2012_dataloader(logger, args, split: str, label_dir: str, task_key: str, client_id=-1, **kwargs):
    logger.info(f"VOC2012: Building dataloader for split='{split}', task='{task_key}'")

    shuffle = True if "train" in split else False
    image_size = getattr(args, "image_size", (224, 224))
    augmentation = getattr(args, "img_augmentation", False)

    dataset = VOC2012_ImagesDataset(
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
