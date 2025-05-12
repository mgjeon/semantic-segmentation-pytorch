# Convert CamVid format to train, val, test dataset with binary masks

"""
data_dir
├── default
│   ├── <image-file1>.png
│   ├── <image-file2>.png
│   └── ...
└── defaultannot
    ├── <annot-file1>.png
    ├── <annot-file2>.png
    └── ...


↓↓↓ Convert to ↓↓↓

dataset_dir
├── train
│   ├── images
│   │   ├── <train-file1>.png
│   │   ├── <train-file2>.png
│   │   └── ...
│   └── masks
│       ├── <train-file1>.png
│       ├── <train-file2>.png
│       └── ...
├── val
│    ├── images
│    │   ├── <val-file1>.png
│    │   ├── <val-file2>.png
│    │   └── ...
│    └── masks
│        ├── <val-file1>.png
│        ├── <val-file2>.png
│        └── ...
└── test
     ├── images
     │   ├── <test-file1>.png
     │   ├── <test-file2>.png
     │   └── ...
     └── masks
         ├── <test-file1>.png
         ├── <test-file2>.png
         └── ...
"""

import warnings
warnings.filterwarnings("ignore")

import shutil
import yaml
import random
import logging
from pathlib import Path
from time import perf_counter

import skimage
import numpy as np
from tqdm import tqdm
from PIL import Image

def main(args):
    log_file = 'create_dataset.txt'
    logging.basicConfig(filename=log_file, 
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(console_handler)

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    start_time = perf_counter()
    
    ########################################
    data_dir = Path(config["data_dir"])

    image_dir = data_dir / "default"
    label_dir = data_dir / "defaultannot"
    clip_dir = data_dir / "contrast"

    images = list(image_dir.glob("*.png"))
    labels = list(label_dir.glob("*.png"))
    clips = list(clip_dir.glob("*.png"))

    assert len(images) == len(labels), "Number of images and labels must be the same."

    train_val_test = config["train_val_test"]
    assert sum(train_val_test) == 1, "The sum of train, val, test must be 1."

    # Shuffle & Split dataset
    n = len(images)
    train = int(n * train_val_test[0])
    val = int(n * train_val_test[1])

    np.random.seed(0)
    random.shuffle(images)

    train_images = images[:train]
    val_images = images[train:train + val]
    test_images = images[train + val:]

    train_labels = [label_dir / (image.name) for image in train_images]
    val_labels = [label_dir / (image.name) for image in val_images]
    test_labels = [label_dir / (image.name) for image in test_images]

    test_clips = [clip_dir / (image.name) for image in test_images]
    
    logging.info("Total : {}".format(len(images)))
    logging.info("Train : {}".format(len(train_images)))
    logging.info("Val   : {}".format(len(val_images)))
    logging.info("Test  : {}".format(len(test_images)))

    # Create dataset directory

    dataset_dir = config["dataset_dir"]
    dataset_dir = Path(dataset_dir)
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    train_image_dir = dataset_dir / "train" / "images"
    train_label_dir = dataset_dir / "train" / "masks"
    train_image_dir.mkdir(parents=True, exist_ok=True)
    train_label_dir.mkdir(parents=True, exist_ok=True)

    val_image_dir = dataset_dir / "val" / "images"
    val_label_dir = dataset_dir / "val" / "masks"
    val_image_dir.mkdir(parents=True, exist_ok=True)
    val_label_dir.mkdir(parents=True, exist_ok=True)

    test_image_dir = dataset_dir / "test" / "images"
    test_label_dir = dataset_dir / "test" / "masks"
    test_image_dir.mkdir(parents=True, exist_ok=True)
    test_label_dir.mkdir(parents=True, exist_ok=True)

    # Copy images and labels
    for image, label in tqdm(zip(train_images, train_labels),
            total=len(train_images),
            desc="Copying images and masks (train)"
        ):
        shutil.copy(image, train_image_dir / image.name)
        # Make binary mask 
        mask = np.array(Image.open(label).convert("RGB"))
        mask = mask.sum(-1).astype(bool).astype(np.uint8)
        Image.fromarray(mask).save(train_label_dir / label.name)

    for image, label in tqdm(zip(val_images, val_labels),
            total=len(val_images),
            desc="Copying images and masks   (val)"
        ):
        shutil.copy(image, val_image_dir / image.name)
        # Make binary mask 
        mask = np.array(Image.open(label).convert("RGB"))
        mask = mask.sum(-1).astype(bool).astype(np.uint8)
        Image.fromarray(mask).save(val_label_dir / label.name)

    for image, label in tqdm(zip(test_images, test_labels),
            total=len(test_images),
            desc="Copying images and masks  (test)"
        ):
        shutil.copy(image, test_image_dir / image.name)
        # Make binary mask 
        mask = np.array(Image.open(label).convert("RGB"))
        mask = mask.sum(-1).astype(bool).astype(np.uint8)
        Image.fromarray(mask).save(test_label_dir / label.name)
    
    ########################################

    end_time = perf_counter()
    logging.info(f"Elapsed time: {end_time - start_time:5.2f} s")  

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/create_dataset.yaml")
    args = parser.parse_args()
    main(args)