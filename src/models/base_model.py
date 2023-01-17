import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import (
    Compose,
    HorizontalFlip,
    CLAHE,
    HueSaturationValue,
    RandomBrightness,
    RandomContrast,
    RandomGamma,
    OneOf,
    Resize,
    ToFloat,
    ShiftScaleRotate,
    RandomRotate90,
    Flip,
    Cutout,
    Crop,
    Normalize,
)


class BaseModel:
    def __init__(self, config):
        self.model = None
        self.device = config["device"]
        self.config = config

    def iteration(self, data):
        pass

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def binarize(self, outputs, labels, thresh):
        pass

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def train_transform(self):
        img_size = self.config["img_size"]
        transform = Compose(
            [
                Resize(img_size[1], img_size[0], p=1.0),
                # A.RandomCrop(int(0.8 * img_size[1]), int(0.8 * img_size[0]), p=0.6),
                # A.Resize(img_size[1], img_size[0]),
                HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                HueSaturationValue(p=0.6),
                RandomBrightness(p=0.7),
                ShiftScaleRotate(p=0.9),
                RandomGamma(p=0.4),
                # Cutout(p=0.7, num_holes=12, max_h_size=8, max_w_size=8) if "cutout" not in self.config else Cutout(**self.config.cutout),
                Normalize(p=1),
                ToTensorV2(),
            ],
            p=1,
        )
        if "train_transform" in self.config:
            transform = A.from_dict({"transform": self.config["train_transform"]})
        return transform

    def test_transform(self):
        img_size = self.config["img_size"]
        transform = Compose(
            [Resize(img_size[1], img_size[0], p=1.0), Normalize(p=1), ToTensorV2()], p=1
        )
        if "test_transform" in self.config:
            transform = A.from_dict({"transform": self.config["test_transform"]})
        return transform

    def get_transform_dicts(self):
        train_tr = self.train_transform()
        test_tr = self.test_transform()
        return A.to_dict(train_tr), A.to_dict(test_tr)