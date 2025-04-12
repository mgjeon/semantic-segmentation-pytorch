"""
Download the balloon dataset from https://github.com/matterport/Mask_RCNN and transform it into CamVid format for binary semantic segmentation.

data_dir
├── default
│   ├── <image-file1>.png
│   ├── <image-file2>.png
│   └── ...
└── defaultannot
    ├── <annot-file1>.png
    ├── <annot-file2>.png
    └── ...
"""


import requests
import zipfile
import shutil

url = "https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip"
filename = "balloon_dataset.zip"

response = requests.get(url)
with open(filename, "wb") as file:
    file.write(response.content)


with zipfile.ZipFile(filename, "r") as file:
    file.extractall()

shutil.rmtree("__MACOSX")


import json
from pathlib import Path
import numpy as np
from PIL import Image
import skimage.draw
from tqdm import tqdm

def create_mask(dataset_dir, new_dataset_dir):
    dataset_dir = Path(dataset_dir)

    img_dir = Path(new_dataset_dir) / "default"
    mask_dir = Path(new_dataset_dir) / "defaultannot"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    with open(dataset_dir / "via_region_data.json") as file:
        annotations = json.load(file)

    for idx, v in tqdm(enumerate(annotations.values()), total=len(annotations)):
        img = np.array(Image.open(dataset_dir / v["filename"]).convert("RGB"))
        height, width = img.shape[:2]

        regions = v["regions"]

        mask = np.zeros([height, width], dtype=np.uint8)
        for region in regions.values():
            anno = region["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = np.array([[y, x] for x, y in zip(px, py)])
            mask += skimage.draw.polygon2mask((height, width), poly)
        mask = mask.astype(np.bool).astype(np.uint8)

        Image.fromarray(img).save(img_dir / (v["filename"][:-4] + ".png"))
        Image.fromarray(mask).save(mask_dir / (v["filename"][:-4] + ".png"))

create_mask("balloon/train", "data")
create_mask("balloon/val", "data")