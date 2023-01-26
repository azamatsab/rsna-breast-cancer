import logging

import numpy as np
import torch
import torchvision.transforms.functional as TF
from timm.models import create_model
from ema_pytorch import EMA
import albumentations as A
from albumentations.pytorch import ToTensorV2

import src
from src.losses import FocalLoss
from src.optimizers import CosineAnnealingWarmupRestarts
from src.models.base_model import BaseModel
from src.mixup import mixup, cutmix, cutmix_criterion


class TimmModel(BaseModel):
    def __init__(self, config):
        model_name = config["experiment_name"]
        self.model = create_model(
            model_name=model_name,
            pretrained=True,
            num_classes=1,
            # in_chans=24,
            drop_rate=0.8,
        )
        self.model = torch.nn.DataParallel(self.model)
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config.pos_weight])).cuda()
        # self.criterion = FocalLoss(logits=True)
        optimizer = getattr(torch.optim, config["optimizer"])
        self.optimizer = optimizer(self.model.parameters(), **config["opt_params"])

        self.device = torch.device(config["device"])
        self.model.to(self.device)
        self.config = config
        self.scheduler = None

        # self.ema = EMA(
        #                 self.model,
        #                 beta = 0.99,              # exponential moving average factor
        #                 update_after_step = 1,    # only after this number of .update() calls will it start updating
        #                 update_every = 1,          # how often to actually update, to save on compute (updates every 10th .update() call)
        #             )

    def set_scheduler(self, length):
        if "first_cycle_steps" in self.config.sch_params:
            self.config.sch_params.first_cycle_steps *= length
            logging.info(f"first_cycle_steps set to {self.config.sch_params.first_cycle_steps}")
        elif "total_steps" in self.config.sch_params:
            self.config.sch_params.total_steps *= length
            logging.info(f"total_steps set to {self.config.sch_params.total_steps}")
        try:
            scheduler = getattr(torch.optim.lr_scheduler, self.config.scheduler)
        except:
            scheduler = getattr(src.optimizers, self.config.scheduler)

        self.scheduler = scheduler(self.optimizer, **self.config.sch_params)

    def ema_update(self):
        self.ema.update()

    def iteration(self, data, train=False, ema=False):
        img, labels = data["img"], data["target"]
        inputs = img.to(self.device, dtype=torch.float)
        labels = labels.to(self.device, dtype=torch.float)

        cutmixed = False
        mixedup = False
        if train: 
            if self.config.mixup_alpha > 0.0:
                if np.random.uniform(0, 1) < self.config.mixup_prob:
                    inputs, labels = mixup(
                        inputs,
                        labels,
                        np.random.beta(
                            self.config.mixup_alpha, self.config.mixup_alpha
                        ),
                    )
                    mixedup = True
            if not mixedup and np.random.uniform(0, 1) < self.config.cutmix_prob:
                cutmixed = True
                inputs, lam, labels_shuffled = cutmix(inputs, labels, self.config.img_size)

        if not ema:
            outputs = self.model(inputs)
        else:
            outputs = self.ema(inputs)
        if cutmixed:
            loss = cutmix_criterion(outputs, labels.unsqueeze(1), 
                            lam, labels_shuffled.unsqueeze(1), self.criterion)
        else:
            loss = self.criterion(outputs, labels.unsqueeze(1))
        return loss, outputs

    def binarize(self, outputs, labels, thresh):
        pred = torch.sigmoid(outputs).data.cpu().numpy()
        labels = labels.cpu().numpy().astype(int)
        return pred, labels

    def tta(self, data):
        img = data["img"]
        hflip = TF.hflip(img)
        vflip = TF.vflip(img)

        loss, outputs = self.iteration(data, False, False)

        data["img"] = hflip
        loss, outputs_h = self.iteration(data, False, False)

        # data["img"] = vflip
        # loss, outputs_v = self.iteration(data, False, False)
        # return loss, (outputs + outputs_h + outputs_v) / 3
        return loss, (outputs + outputs_h) / 2
        

    # def train_transform(self):
    #     img_size = self.config["img_size"]
    #     transform = A.Compose([
    #             A.Resize(img_size[1], img_size[0]),
    #             # A.RandomCrop(int(0.8 * img_size[1]), int(0.8 * img_size[0]), p=0.6),
    #             # A.Resize(img_size[1], img_size[0]),
    #             A.RandomBrightnessContrast(p=0.4, brightness_limit=0.25, contrast_limit=0.25),
    #             A.HorizontalFlip(p=0.5),
    #             A.OneOf([
    #                 A.CLAHE(),
    #                 A.RandomGamma()
    #                 ], p=1.0
    #             ),
    #             A.ShiftScaleRotate(rotate_limit=15, scale_limit=0.2, shift_limit=0.1, p=0.9),
    #             A.OneOf([
    #                 A.Blur(blur_limit=7, p=1.0),
    #                 A.MotionBlur(),
    #                 A.GaussNoise(),
    #                 A.ImageCompression(quality_lower=75)
    #             ], p=0.5),

    #             A.Cutout(num_holes=64, max_h_size=16, max_w_size=16, fill_value=250, p=0.65) if "cutout" not in self.config else A.Cutout(**self.config.cutout),
    #             A.Normalize(),
    #             ToTensorV2(),
    #             ]
    #         )
    #     if "train_transform" in self.config:
    #         transform = A.from_dict({"transform": self.config["train_transform"]})
    #     return transform

    # def test_transform(self):
    #     img_size = self.config["img_size"]
    #     transform = A.Compose(
    #         [A.Resize(img_size[1], img_size[0], p=1.0), Normalize(p=1), ToTensorV2()], p=1
    #     )
    #     if "test_transform" in self.config:
    #         transform = A.from_dict({"transform": self.config["test_transform"]})
    #     return transform
