from train.multi_label_tasks.train_multi_label_cls import MultiLabelTrainer

cifar10_train_config = {
    "task_name": "cifar10",
    "images_source": "cifar10",
    "splits": ["train", "val_small"],
    "classifier_type": "FC_Classifier",
    "num_labels": 10,
    "num_images": 1,
    "model_type": "classification",
    "num_epochs": 20,
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.1,
    "task_trainer": MultiLabelTrainer,
    "random_baseline_score": 0.0,
}

voc2012_train_config = {
    "task_name": "voc2012",
    "images_source": "voc2012",
    "splits": ["train", "val"],
    "classifier_type": "FC_Classifier",
    "num_labels": 20,
    "num_images": 1,
    "model_type": "classification",
    "num_epochs": 20,
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.1,
    "task_trainer": MultiLabelTrainer,
    "random_baseline_score": 0.0,
}

coco2017_train_config = {
    "task_name": "coco2017",
    "images_source": "coco2017",
    "splits": ["train", "val"],
    "classifier_type": "FC_Classifier",
    "num_labels": 80,
    "num_images": 1,
    "model_type": "classification",
    "num_epochs": 20,
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.1,
    "task_trainer": MultiLabelTrainer,
    "random_baseline_score": 0.0,
}

xray14_train_config = {
    "task_name": "xray14",
    "images_source": "xray14",
    "splits": ["train", "val"],
    "classifier_type": "FC_Classifier",
    "num_labels": 15,
    "num_images": 1,
    "model_type": "classification",
    "num_epochs": 20,
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.1,
    "task_trainer": MultiLabelTrainer,
    "random_baseline_score": 0.0,
}

medmnist_train_config = {
    "task_name": "medmnist",
    "images_source": "medmnist",
    "splits": ["train", "val_small"],
    "classifier_type": "FC_Classifier",
    "num_labels": 10,
    "num_images": 1,
    "model_type": "classification",
    "num_epochs": 20,
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.1,
    "task_trainer": MultiLabelTrainer,
    "random_baseline_score": 0.0,
}

task_configs = {
    "cifar10_train": cifar10_train_config,
    "medmnist_train": medmnist_train_config,
    "voc2012_train": voc2012_train_config,
    "xray14_train": xray14_train_config,
    "coco2017_train": coco2017_train_config
}
