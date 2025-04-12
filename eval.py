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

from torch.utils.data import DataLoader, SequentialSampler

from utils.model import get_model
from utils.data import BinaryDataset
from train import evaluate, collate_fn

import os

from mpl_toolkits.axes_grid1 import make_axes_locatable

def main(config):
    log_file = 'eval.txt'
    logging.basicConfig(filename=log_file, 
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(console_handler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = Path(config['eval']["dataset_path"])
    logging.info(f"Dataset path: {str(dataset_path)}")

    dataset_test = BinaryDataset(dataset_path, stage="test")

    image_path = dataset_path / "images"
    mask_path = dataset_path / "masks"

    images = sorted(image_path.glob("*.png"))
    logging.info(f"Number of images: {len(images)}")

    model = get_model(name=config['model']["name"],
                      weights=None,
                      weights_backbone=None,
                      aux_loss=config['model']["aux_loss"],
                      num_classes=2)
    
    model.load_state_dict(torch.load(config['eval']["checkpoint"], map_location=device, weights_only=True))
    model = model.to(device)
    model = model.eval()

    preprocess = dataset_test.transform

    p = config['eval']["probability_threshold"]
    alpha = config['eval']["alpha"]
    color = config['eval']["color"]

    result_dir = Path(config['eval']["result_dir"])
    result_dir.mkdir(exist_ok=True, parents=True)

    data_loader_test = DataLoader(
        dataset_test,
        batch_size=1,
        sampler=SequentialSampler(dataset_test),
        num_workers=config['train']["num_workers"],
        drop_last=False,
        collate_fn=collate_fn
    )

    acc_global, acc, iou = evaluate(model, data_loader_test, 
                                    device, num_classes=config['model']["num_classes"])
    
    logging.info("overall   accuracy : {:.1f}".format(acc_global.item() * 100))
    logging.info("per-class accuracy : {}".format([f"{i:.1f}" for i in (acc * 100).tolist()]))
    logging.info("per-class IoU      : {}".format([f"{i:.1f}" for i in (iou * 100).tolist()]))
    mean_iou = iou.mean().item() * 100
    logging.info("mean IoU           : {:.1f}".format(mean_iou))

    eval_result = result_dir / "eval_result.txt"
    with open(eval_result, "w") as file:
        file.write(f"model path         : {os.path.abspath(config['eval']['checkpoint'])}\n")
        file.write(f"data  path         : {os.path.abspath(dataset_path)}\n")
        file.write(f"overall   accuracy : {acc_global.item() * 100:.1f}\n")
        file.write(f"per-class accuracy : {', '.join([f'{i:.1f}' for i in (acc * 100).tolist()])}\n")
        file.write(f"per-class IoU      : {', '.join([f'{i:.1f}' for i in (iou * 100).tolist()])}\n")
        file.write(f"mean IoU           : {mean_iou:.1f}\n")


    for file in tqdm(images, desc="Evaluation"):
        mask_file = mask_path / file.name

        mask = decode_image(mask_file)
        mask = mask[0].to(torch.bool)

        img = Image.open(file)
        img2 = decode_image(file)
        w, h = img.size

        image = preprocess(img)
        image = image.unsqueeze(0).to(device)


        outputs = model(image)
        output = outputs['out']
        probs = output.softmax(1)

        prop = probs[0, 1].detach().cpu()
        prop = v2.Resize(size=(h, w))(prop.unsqueeze(0))
        prop = prop[0]

        true_mask = v2.Resize(size=(h, w))(mask.unsqueeze(0))
        true_mask = true_mask[0].to(torch.bool)
        img_truemask = draw_segmentation_masks(img2, masks=true_mask, 
                                               alpha=alpha, colors=[color])
        img_truemask = img_truemask.permute(1, 2, 0)
        
        pred_mask = (prop > p).to(torch.bool)
        img_predmask = draw_segmentation_masks(img2, masks=pred_mask, 
                                               alpha=alpha, colors=[color])
        img_predmask = img_predmask.permute(1, 2, 0)

        fig = plt.figure(figsize=config['eval']['figsize'])
        gs = fig.add_gridspec(3, 3)  # subplot 3x3 기준

        ax = fig.add_subplot(gs[0])
        ax.imshow(img)
        ax.set_title("Input")
        ax.axis("off")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cax.axis("off")

        ax = fig.add_subplot(gs[1])
        ax.imshow(true_mask, cmap=cmp)
        ax.set_title("True Mask")
        ax.axis("off")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cax.axis("off")

        ax = fig.add_subplot(gs[2])
        ax.imshow(pred_mask, cmap=cmp)
        ax.set_title(f"Pred Mask (p > {p})")
        ax.axis("off")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cax.axis("off")

        ax = fig.add_subplot(gs[3])
        ax.axis("off")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cax.axis("off")

        ax = fig.add_subplot(gs[4])
        ax.imshow(img_truemask)
        ax.set_title("Overlay (True)")
        ax.axis("off")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cax.axis("off")

        ax = fig.add_subplot(gs[5])
        ax.imshow(img_predmask)
        ax.set_title("Overlay (Pred)")
        ax.axis("off")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cax.axis("off")

        ax = fig.add_subplot(gs[6])
        ax.axis("off")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cax.axis("off")
        
        ax = fig.add_subplot(gs[7])
        ax.axis("off")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cax.axis("off")

        ax = fig.add_subplot(gs[8])
        im = ax.imshow(prop, clim=(0, 1))
        ax.set_title("Probability Map")
        ax.axis("off")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        

        fig.suptitle(f"{file.stem}")
        plt.tight_layout()
        plt.savefig(result_dir / file.name, dpi=300)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/main.yaml")
    parser.add_argument("--stage", type=str, default="test")
    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    main(config)