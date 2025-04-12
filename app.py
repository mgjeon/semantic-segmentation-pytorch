from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmp=ListedColormap(['black','white'])
plt.rcParams.update({'font.size': 20})

import torch
from torchvision.io import read_image
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.v2 as v2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from utils.model import get_model
from utils.data import BinaryDatasetWithoutLabel

import gradio as gr

# [Function for gradio] ====================================================================
def predict(checkpoint, image_file, p, alpha, color):
    if color[0] != "#":
        rgba_values = color[5:-1].split(',')
        color = tuple(int(float(value)) for value in rgba_values[:3])
    
    preprocess = BinaryDatasetWithoutLabel("")._get_transforms()

    model = get_model(name="fcn_resnet50",
                    weights=None,
                    weights_backbone=None,
                    aux_loss=True,
                    num_classes=2)

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model = model.to(device)
    model = model.eval()

    img_ori = read_image(image_file)

    img = Image.open(image_file)
    w, h = img.size

    image = preprocess(img)
    image = image.unsqueeze(0).to(device)

    outputs = model(image)
    output = outputs['out']
    probs = output.softmax(1)

    prob = probs[0, 1].detach().cpu()
    prob = v2.Resize(size=(h, w))(prob.unsqueeze(0))
    prob = prob[0]

    pred_mask = (prob > p).to(torch.bool)

    img_predmask = draw_segmentation_masks(img_ori, masks=pred_mask,
                                            alpha=alpha, colors=[color])
    img_predmask = img_predmask.permute(1, 2, 0)

    fig_prob = plt.figure()
    ax = fig_prob.add_subplot(111)
    ax.imshow(prob, clim=(0, 1), cmap="viridis")
    ax.axis("off")
    fig_prob.tight_layout()

    fig_pred_mask = plt.figure()
    ax = fig_pred_mask.add_subplot(111)
    ax.imshow(pred_mask, cmap=cmp)
    ax.axis("off")
    fig_pred_mask.tight_layout()

    fig_overlay_raw = plt.figure()
    ax = fig_overlay_raw.add_subplot(111)
    ax.imshow(img_predmask)
    ax.axis("off")
    fig_overlay_raw.tight_layout()

    return fig_prob, fig_pred_mask, fig_overlay_raw
# [Function for gradio] ====================================================================

if __name__ == "__main__":
    demo = gr.Interface(fn=predict,
                        inputs=[
                            gr.File(label="Model Checkpoint", type="filepath"),
                            gr.Image(label="Raw Image (Input)", type="filepath"),
                            gr.Number(0.5, label="Probability Threshold (0-1)"),
                            gr.Number(0.5, label="Overlay Opacity (0-1)"),
                            gr.ColorPicker("#ff0000", label="Overlay Color"),
                        ], 
                        outputs=[
                            gr.Plot(label="Probability Map (Output)"),
                            gr.Plot(label="Segmentation Mask (Output)"),
                            gr.Plot(label="Overlay over Raw Image (Output)"),
                        ],)
    demo.launch()