import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

from data.image_datasets.data_augmentation import transforms_train_augmented


# ==============================================================
# ======================= COCO 2017 Classes ====================
# ==============================================================
# Standard 80 COCO instance categories (names). Keep exactly 80.
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
    "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
    "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant",
    "bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors",
    "teddy bear","hair drier","toothbrush"
]
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(COCO_CLASSES)}


# ==============================================================
# ======================= COCO Annotation Parse =================
# ==============================================================
def load_coco_instances_index(ann_json_path: str, use_supercategory: bool = False):
    """
    Builds fast lookup structures from COCO instances_*.json.

    Returns:
        img_id_to_file: dict[int,str]
        img_id_to_multihot: dict[int, torch.FloatTensor]  # [80]
        cat_id_to_name: dict[int,str]  # chosen label name
        name_to_idx: dict[str,int]     # label -> 0..79
    """
    with open(ann_json_path, "r") as f:
        coco = json.load(f)

    # category id -> name (or supercategory)
    cat_id_to_name = {}
    for c in coco["categories"]:
        name = c["supercategory"] if use_supercategory else c["name"]
        cat_id_to_name[c["id"]] = name

    # validate label space (we want the 80 names above)
    # If your JSON uses the standard categories, this should match.
    # If you used use_supercategory=True, label space is smaller; update COCO_CLASSES accordingly.
    name_to_idx = CLASS_TO_IDX

    # image_id -> file_name
    img_id_to_file = {im["id"]: im["file_name"] for im in coco["images"]}

    # image_id -> set(label_names)
    img_id_to_labels = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        name = cat_id_to_name.get(cat_id, None)
        if name is None:
            continue
        if name not in name_to_idx:
            # Skip anything outside the 80-class list
            continue
        img_id_to_labels.setdefault(img_id, set()).add(name)

    # image_id -> multi-hot tensor
    img_id_to_multihot = {}
    for img_id, labels in img_id_to_labels.items():
        vec = torch.zeros(len(COCO_CLASSES), dtype=torch.float32)
        for lbl in labels:
            vec[name_to_idx[lbl]] = 1.0
        img_id_to_multihot[img_id] = vec

    return img_id_to_file, img_id_to_multihot, cat_id_to_name, name_to_idx


def multi_hot_to_class_indices(label_tensor, threshold=0.0):
    return (label_tensor > threshold).nonzero(as_tuple=False).flatten().tolist()


# ==============================================================
# ======================= COCO Dataset ==========================
# ==============================================================
class COCO2017_ImagesDataset(Dataset):
    """
    Follows the same output format as your VOC2012_ImagesDataset:
        returns dict:
          {
            "index": idx,
            "images": image (PIL or tensor depending on your pipeline),
            "labels": list[int]      # active class indices
            "target_scores": tensor  # multi-hot [80]
          }

    IMPORTANT:
    - This assumes your FL split files contain COCO *image_id* integers, one per line.
      e.g. "train_step_0_client_1.txt" lines: 391895
    - Unlabeled images are handled by:
        - dropping them when generating split files, OR
        - keeping them but setting target_scores all-zero (you can choose below)
    """
    def __init__(
        self,
        split: str,
        dataset_dir: str,
        label_dir: str,
        task_key: str,
        image_size=(224, 224),
        img_augmentation=False,
        drop_unlabeled: bool = True,
    ):
        self.image_size = image_size
        self.img_augmentation = img_augmentation
        self.split = split
        self.num_labels = 80

        # Your dataset_dir should contain coco-2017 (as in your output path)
        # structure expected:
        #   dataset_dir/
        #     coco-2017/
        #       train2017/
        #       val2017/
        #       annotations/instances_train2017.json, instances_val2017.json
        self.base_dir = os.path.join(dataset_dir, "coco-2017")

        # Map split string to COCO folder/json naming
        # Accept "train" or "train2017"; "val" or "val2017"
        if "train" in split:
            coco_img_folder = "train/data"
            ann_json = "instances_train2017.json"
        else:
            coco_img_folder = "validation/data"
            ann_json = "instances_val2017.json"

        self.image_dir = os.path.join(self.base_dir, coco_img_folder)
        self.ann_json_path = os.path.join(self.base_dir, "annotations", ann_json)

        # Load COCO indices
        self.img_id_to_file, self.img_id_to_multihot, _, _ = load_coco_instances_index(self.ann_json_path)

        # Determine list of image_ids for this dataset split / FL client
        if "train" in split:
            buff = task_key.split("_")
            # match your VOC naming: train_step_{step}_client_{id}
            label_task_key = "train_step_{0}_client_{1}".format(buff[-3], buff[-1])
            data_set_file = os.path.join(self.base_dir, "splits", f"{label_dir}", f"{label_task_key}.txt")
        else:
            # for eval, you might have a val.txt listing COCO image_ids
            data_set_file = os.path.join(self.base_dir, "splits", f"{label_dir}", "val.txt")

        with open(data_set_file, "r") as f:
            # COCO image ids are ints; your saved files write them per line
            self.image_ids = [int(x.strip()) for x in f.readlines() if x.strip()]

        # Optionally drop unlabeled ids (so you stay in 80 classes only, no "Others")
        if drop_unlabeled:
            self.image_ids = [i for i in self.image_ids if i in self.img_id_to_multihot]

        self.raw_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # Resolve file name
        file_name = self.img_id_to_file.get(image_id, None)
        if file_name is None:
            raise FileNotFoundError(f"COCO image_id {image_id} not found in annotations images list")

        img_path = os.path.join(self.image_dir, file_name)
        image = Image.open(img_path).convert("RGB")

        if self.img_augmentation and ("train" in self.split):
            image = transforms_train_augmented(image)

        # Multi-label target
        if image_id in self.img_id_to_multihot:
            target_scores = self.img_id_to_multihot[image_id]
        else:
            # unlabeled image: either dropped earlier or represented as all-zero
            target_scores = torch.zeros(len(COCO_CLASSES), dtype=torch.float32)

        labels = multi_hot_to_class_indices(target_scores)

        return {
            "index": idx,
            "images": image,
            "labels": labels,
            "target_scores": target_scores
        }

    def get_label_distribution(self):
        """
        Computes label counts over this dataset's image_ids by summing multi-hot vectors.
        """
        label_counts = np.zeros(len(COCO_CLASSES), dtype=np.int64)
        for img_id in self.image_ids:
            vec = self.img_id_to_multihot.get(img_id, None)
            if vec is None:
                continue
            label_counts += vec.numpy().astype(np.int64)
        return label_counts


# ==============================================================
# ======================= Collate + Builder =====================
# ==============================================================
def resnet_batch_collate(batch):
    images = [item["images"] for item in batch]
    labels = [item["labels"] for item in batch]
    target_scores = [item["target_scores"] for item in batch]
    indices = [item["index"] for item in batch]

    return {
        "index": torch.tensor(indices, dtype=torch.long),
        "images": images,
        "labels": labels,
        "target_scores": torch.stack(target_scores)
    }


def build_coco2017_dataloader(logger, args, split: str, label_dir: str, task_key: str, client_id=-1, **kwargs):
    logger.info(f"COCO2017: Building dataloader for split='{split}', task='{task_key}'")

    shuffle = True if "train" in split else False
    image_size = getattr(args, "image_size", (224, 224))
    augmentation = getattr(args, "img_augmentation", False)

    dataset = COCO2017_ImagesDataset(
        split=split,
        dataset_dir=args.data_dir,      # parent dir containing coco-2017/
        label_dir=label_dir,            # your split folder name
        task_key=task_key,
        image_size=image_size,
        img_augmentation=augmentation,
        drop_unlabeled=True,            # keeps only real 80 classes
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
