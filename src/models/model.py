import logging

import numpy as np
import torch
from efficientnet_pytorch import EfficientNet
import albumentations as A
from albumentations.pytorch import ToTensorV2

import src
from src.losses import FocalLoss
from src.optimizers import CosineAnnealingWarmupRestarts
from src.models.base_model import BaseModel
from src.mixup import mixup, cutmix, cutmix_criterion
from src.utils import freeze


class GenModel(BaseModel):
    def __init__(self, config):
        model_name = config["experiment_name"]
        self.model = EfficientNet.from_name(model_name, num_classes=1)
        freeze(self.model, "_blocks.20._expand_conv.weight")
        self.model = torch.nn.DataParallel(self.model)
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config.pos_weight])).cuda()
        # self.criterion = FocalLoss(logits=True)
        optimizer = getattr(torch.optim, config["optimizer"])
        self.optimizer = optimizer(self.model.parameters(), **config["opt_params"])

        try:
            scheduler = getattr(torch.optim.lr_scheduler, config["scheduler"])
        except:
            scheduler = getattr(src.optimizers, config["scheduler"])

        self.scheduler = scheduler(self.optimizer, **config["sch_params"])
        self.device = torch.device(config["device"])
        self.model.to(self.device)
        self.config = config

    def load(self, path):
        logging.info("Loading pretrained weights")
        self.model.load_state_dict(torch.load(path), strict=False)

    def iteration(self, data, train=False):
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

        outputs = self.model(inputs)
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
