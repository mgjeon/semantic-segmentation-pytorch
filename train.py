import yaml
import datetime
import logging
from time import perf_counter
from pathlib import Path

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from utils.data import BinaryDataset
from utils.metrics import ConfusionMatrix
from utils.model import get_model


def criterion(outputs, target):
    """
    outputs: {"out": [batch_size, num_classes, H, W], "aux": [batch_size, num_classes, H, W]}
    target : [batch_size, M, M]
    """
    losses = {}
    for name, output in outputs.items():
        losses[name] = torch.nn.functional.cross_entropy(output, target, ignore_index=255)

    return losses["out"] + 0.5 * losses["aux"]


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    
    confmat = ConfusionMatrix(num_classes)

    with torch.inference_mode():
        for image, target in data_loader:
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output["out"]

            true_mask = target.flatten()
            pred_mask = output.argmax(1).flatten()

            confmat.update(true_mask, pred_mask)

    acc_global, acc, iou = confmat.compute()
    return acc_global, acc, iou


def collate_fn(batch):
    images = torch.stack([sample["image"] for sample in batch])
    masks = torch.stack([sample["mask"] for sample in batch])
    return images, masks

def collate_fn_withoutmask(batch):
    images = torch.stack([sample["image"] for sample in batch])
    return images


def get_dataloaders(config):
    dataset_root = Path(config['train']["dataset_root"])
    dataset_train_dir = dataset_root / "train"
    dataset_val_dir = dataset_root / "val"

    dataset_train = BinaryDataset(dataset_train_dir, stage="train")
    dataset_val = BinaryDataset(dataset_val_dir, stage="val")

    batch_size = config['train']["batch_size"]
    num_workers = config['train']["num_workers"]

    data_loader_train = DataLoader(dataset_train, 
                                   batch_size=batch_size, 
                                   sampler=RandomSampler(dataset_train),
                                   num_workers=num_workers, 
                                   drop_last=True,
                                   collate_fn=collate_fn)

    data_loader_val = DataLoader(dataset_val,
                                 batch_size=1,
                                 sampler=SequentialSampler(dataset_val),
                                 num_workers=num_workers,
                                 drop_last=False,
                                 collate_fn=collate_fn)
    
    return data_loader_train, data_loader_val


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        func(*args, **kwargs)
        total_time = perf_counter() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info(f"Training time {total_time_str}")
    return wrapper


@timer
def main(config):
    log_file = 'train.txt'
    logging.basicConfig(filename=log_file, 
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(console_handler)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_classes = 2  # Aurora dataset has 2 classes (background, aurora)

    data_loader_train, data_loader_val = get_dataloaders(config)
    weights = config['model']["weights"]
    weights_backbone = config['model']["weights_backbone"]
    if weights == "None": weights = None
    if weights_backbone == "None": weights_backbone = None
    logging.info(f"weights          : {weights}")
    logging.info(f"weights_backbone : {weights_backbone}")
    model = get_model(name=config['model']["name"], 
                      weights=weights,
                      weights_backbone=weights_backbone,
                      aux_loss=config['model']["aux_loss"],
                      num_classes=num_classes)
    model.to(device)

    params = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model.classifier.parameters() if p.requires_grad]},
        {"params": [p for p in model.aux_classifier.parameters() if p.requires_grad]},
    ]

    optimizer = torch.optim.SGD(params, 
                                lr=config['train']["lr"], 
                                momentum=config['train']["momentum"], 
                                weight_decay=config['train']["weight_decay"])
    
    n_epochs = config['train']["n_epochs"]
    iters_per_epoch = len(data_loader_train)
    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer, 
        total_iters=iters_per_epoch * n_epochs, 
        power=0.9
    )

    result_dir = Path(config['train']["result_dir"])
    result_dir.mkdir(exist_ok=True, parents=True)

    mean_iou_best = 0
    epoch_best = 0
    epoch_start = 0

    checkpoint_path = result_dir / "checkpoint.pth"
    if config['train']['resume'] and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        mean_iou_best = checkpoint["mean_iou_best"]
        epoch_best = checkpoint["epoch_best"]
        epoch_start = checkpoint["epoch"] + 1
        logging.info(f"Resuming training from epoch {epoch_start}")

    for epoch in range(epoch_start, n_epochs):
        
        # Training
        model.train()
        
        tqdm_train = tqdm(data_loader_train)
        tqdm_train.set_description("Epoch {:>4d}/{:4d} (Train)".format(epoch+1, n_epochs))
        train_loss = []
        for batch in tqdm_train:
            image, target = batch
            image, target = image.to(device), target.to(device)

            output = model(image)
            loss = criterion(output, target)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_scheduler.step()

            tqdm_train.set_postfix_str("loss {:.4f}, lr {:.4f}".format(loss.item(), optimizer.param_groups[0]["lr"]))
            train_loss.append(loss.item())
        
        train_loss_avg = sum(train_loss) / len(train_loss)
            
        # Validation
        acc_global, acc, iou = evaluate(model, data_loader_val, device, num_classes)
        
        logging.info("Epoch {:>4d}/{:4d} (Val)".format(epoch+1, n_epochs))
        logging.info("overall   accuracy : {:.1f}".format(acc_global.item() * 100))
        logging.info("per-class accuracy : {}".format([f"{i:.1f}" for i in (acc * 100).tolist()]))
        logging.info("per-class IoU      : {}".format([f"{i:.1f}" for i in (iou * 100).tolist()]))
        mean_iou = iou.mean().item() * 100
        logging.info("mean IoU           : {:.1f}".format(mean_iou))
        logging.info("train loss         : {:.4f}".format(train_loss_avg))

        if (epoch+1) % config['train']["save_freq"] == 0:
            torch.save(model.state_dict(), result_dir / f"model_{epoch+1}.pth")
            logging.info(f"Saved model_{epoch+1}.pth")

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "loss": train_loss_avg,
            "epoch": epoch,
            "mean_iou_best": mean_iou_best,
            "epoch_best": epoch_best
        }
        torch.save(checkpoint, checkpoint_path)

        if mean_iou > mean_iou_best:
            epoch_best = epoch
            mean_iou_best = mean_iou
            torch.save(model.state_dict(), result_dir / "model_best.pth")

    logging.info("Best mean IoU: {:.1f} at Epoch {:4d}".format(mean_iou_best, epoch_best+1))
    model_best_txt = result_dir / "model_best.txt"
    with open(model_best_txt, "w") as file:
        file.write(f"Best mean IoU: {mean_iou_best:.1f} at Epoch {epoch_best+1}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/main.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    main(config)
