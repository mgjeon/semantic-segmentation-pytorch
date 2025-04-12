from pathlib import Path

import torch
from torch.utils.data import Dataset

from torchvision.transforms import v2
from torchvision import tv_tensors
from torchvision.io import read_image


class BinaryDataset(Dataset):
    def __init__(
            self, 
            dataset_dir, 
            stage = "train",
            base_size = 520,
            crop_size = 480,
            hflip_prob = 0.5,
            mean = (0.485, 0.456, 0.406), # ImageNet
            std  = (0.229, 0.224, 0.225)  # ImageNet
        ):

        self.stage = stage
        self.base_size = base_size
        self.crop_size = crop_size
        self.hflip_prob = hflip_prob
        self.mean = mean
        self.std = std

        self.dataset_dir = Path(dataset_dir)
        self.transform = self._get_transforms()

        self.img_dir = self.dataset_dir / "images"
        self.mask_dir = self.dataset_dir / "masks"

        self.imgs = list(self.img_dir.glob("*.png"))


    def _get_transforms(self):
        if self.stage == "train":
            transform_train = v2.Compose(
                [
                    v2.ToImage(),
                    v2.Resize(size=(self.base_size, self.base_size)),
                    v2.RandomHorizontalFlip(p=self.hflip_prob),
                    v2.RandomCrop(size=(self.crop_size, self.crop_size)),
                    v2.ToDtype(dtype={tv_tensors.Image:torch.float32, tv_tensors.Mask:torch.int64, "others":None}, scale=True),
                    v2.Normalize(mean=self.mean, std=self.std), 
                    v2.ToPureTensor()
                ]
            ) 
            return transform_train
        elif self.stage == "val" or self.stage == "test":
            transform_val = v2.Compose(
                [
                    v2.ToImage(),
                    v2.Resize(size=self.base_size),
                    v2.ToDtype(dtype={tv_tensors.Image:torch.float32, tv_tensors.Mask:torch.int64, "others":None}, scale=True),
                    v2.Normalize(mean=self.mean, std=self.std),
                    v2.ToPureTensor()
                ]
            )
            return transform_val
        else:
            raise NotImplementedError(f"Stage {self.stage} is not implemented. 'train' or 'val' or 'test' only.")


    def __len__(self):
        return len(self.imgs)


    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        mask_path = self.mask_dir / img_path.name

        img = read_image(img_path)
        mask = read_image(mask_path)[0]

        img = tv_tensors.Image(img)
        mask = tv_tensors.Mask(mask)

        sample = {
            "image": img,
            "mask": mask
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
    

class BinaryDatasetWithoutLabel(Dataset):
    def __init__(
            self, 
            dataset_dir, 
            base_size = 520,
            crop_size = 480,
            hflip_prob = 0.5,
            mean = (0.485, 0.456, 0.406), # ImageNet
            std  = (0.229, 0.224, 0.225)  # ImageNet
        ):

        self.base_size = base_size
        self.crop_size = crop_size
        self.hflip_prob = hflip_prob
        self.mean = mean
        self.std = std

        self.dataset_dir = Path(dataset_dir)
        self.transform = self._get_transforms()

        self.img_dir = self.dataset_dir / "images"

        self.imgs = list(self.img_dir.glob("*.png"))


    def _get_transforms(self):
        transform_val = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(size=self.base_size),
                v2.ToDtype(dtype={tv_tensors.Image:torch.float32}, scale=True),
                v2.Normalize(mean=self.mean, std=self.std),
                v2.ToPureTensor()
            ]
        )
        return transform_val


    def __len__(self):
        return len(self.imgs)


    def __getitem__(self, idx):
        img_path = self.imgs[idx]

        img = read_image(img_path)

        img = tv_tensors.Image(img)

        sample = {
            "image": img,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample