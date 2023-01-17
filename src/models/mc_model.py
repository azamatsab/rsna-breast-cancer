import random
import logging

import numpy as np
import torch
from timm.models import create_model
import albumentations as A
from albumentations.pytorch import ToTensorV2

import src
from src.losses import FocalLoss
from src.optimizers import CosineAnnealingWarmupRestarts
from src.models.base_model import BaseModel
from src.mixup import mixup, cutmix, cutmix_criterion
from src.utils import remove_parallel


class Net(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, img):
        w, h = 256, 256
        if self.training:
            tempr = 4
            step_w = w
            step_h = h
        else:
            step_w = w - 20
            step_h = h - 20
            tempr = 1
        outs = []
        tiles = [img[:, :, x:x+w,y:y+h] for x in range(0, img.shape[2], step_w) for y in range(0, img.shape[3], step_h)]
        random.shuffle(tiles)
        for tile in tiles:
            outs.append(self.backbone(tile) / tempr)
        # for i in range(0, img.shape[1], 3):
        #     outs.append(self.backbone(img[:, i: i + 3]))
        outs = torch.cat(outs, dim=-1)
        return torch.sum(outs, dim=-1).unsqueeze(1)


class MCModel(BaseModel):
    def __init__(self, config):
        model_name = config["experiment_name"]
        backbone = create_model(
            model_name=model_name,
            pretrained=False,
            num_classes=1,
        )
        path = "outputs/tf_efficientnet_b2_4_1/weights/tf_efficientnet_b2_1_0.3828_0.3056.pth"
        backbone.load_state_dict(remove_parallel(torch.load(path)))

        self.model = Net(backbone)

        self.model = torch.nn.DataParallel(self.model)
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config.pos_weight])).cuda()
        # self.criterion = FocalLoss(logits=True)
        optimizer = getattr(torch.optim, config["optimizer"])
        self.optimizer = optimizer(self.model.parameters(), **config["opt_params"])

        self.device = torch.device(config["device"])
        self.model.to(self.device)
        self.config = config

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
