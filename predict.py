import yaml 
import logging
from pathlib import Path
from PIL import Image

from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmp=ListedColormap(['black','white'])
plt.rcParams.update({'font.size': 20})

import torch
from torchvision.io import decode_image
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.v2 as v2

from utils.model import get_model
from utils.data import BinaryDatasetWithoutLabel

from mpl_toolkits.axes_grid1 import make_axes_locatable

def main(config):
    log_file = 'predict.txt'
    logging.basicConfig(filename=log_file, 
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(console_handler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = Path(config['predict']["dataset_path"])
    logging.info(f"Dataset path: {str(dataset_path)}")

    dataset_test = BinaryDatasetWithoutLabel(dataset_path)

    image_path = dataset_path

    images = sorted(image_path.glob("*.png"))
    logging.info(f"Number of images: {len(images)}")

    model = get_model(name=config['model']["name"],
                      weights=None,
                      weights_backbone=None,
                      aux_loss=config['model']["aux_loss"],
                      num_classes=2)
    
    model.load_state_dict(torch.load(config['predict']["checkpoint"], map_location=device, weights_only=True))
    model = model.to(device)
    model = model.eval()

    preprocess = dataset_test.transform

    p = config['predict']["probability_threshold"]
    alpha = config['predict']["alpha"]
    color = config['predict']["color"]

    result_dir = Path(config['predict']["result_dir"])
    result_dir.mkdir(exist_ok=True, parents=True)

    for file in tqdm(images, desc="Prediction"):
        img2 = decode_image(file)

        img = Image.open(file)
        w, h = img.size

        image = preprocess(img)
        image = image.unsqueeze(0).to(device)

        outputs = model(image)
        output = outputs['out']
        probs = output.softmax(1)

        prop = probs[0, 1].detach().cpu()
        prop = v2.Resize(size=(h, w))(prop.unsqueeze(0))
        prop = prop[0]

        pred_mask = (prop > p).to(torch.bool)
        img_predmask = draw_segmentation_masks(img2, masks=pred_mask, 
                                               alpha=alpha, colors=[color])
        img_predmask = img_predmask.permute(1, 2, 0)

        fig, axes = plt.subplots(3, 2, figsize=config['predict']["figsize"])
        axes = axes.ravel()
        ax = axes[0]
        ax.imshow(img)
        ax.set_title("Input")
        ax.axis("off")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cax.axis("off")

        ax = axes[1]
        ax.imshow(pred_mask, cmap=cmp)
        ax.set_title(f"Pred Mask (p > {p})")
        ax.axis("off")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cax.axis("off")

        ax = axes[2]
        ax.axis("off")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cax.axis("off")

        ax = axes[3]
        ax.imshow(img_predmask)
        ax.set_title("Overlay (Pred)")
        ax.axis("off")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cax.axis("off")

        ax = axes[4]
        ax.axis("off")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cax.axis("off")
        
        ax = axes[5]
        im = ax.imshow(prop, clim=(0, 1))
        ax.set_title("Probability Map")
        ax.axis("off")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)

        fig.suptitle(f"{file.stem}")
        plt.tight_layout()
        plt.savefig(result_dir / file.name, dpi=300)
        plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/predict.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    main(config)