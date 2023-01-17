import os

from tqdm import tqdm
import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import cv2


class CamCalculator:
    def __init__(self, config, model, root="cam"):
        self.config = config
        self.root = root
        self.model = model
        model = model.model.module
        model.eval()
        target_layers = [model.conv_head]
        self.cam = GradCAM(model=model, target_layers=target_layers, use_cuda=1)
        self.sigmoid = torch.nn.Sigmoid()
        os.makedirs(self.root, exist_ok=True)

    def calculate_cam(self, loader):
        for data in tqdm(loader):
            input_tensor = data["img"]
            targets = data["target"]
            ctargets = [ClassifierOutputTarget(0) for trg in targets]
            grayscale_cam = self.cam(input_tensor=input_tensor, targets=ctargets)
            _, preds = self.model.iteration(data, False)
            preds = self.sigmoid(preds).detach().cpu().numpy()
            imgs = data["bgr"].numpy()
            for i, cam_img in enumerate(grayscale_cam):
                visualization = show_cam_on_image(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB) / 255, cam_img, use_rgb=True)
                img_name = data["path"][i]
                target = targets[i]
                pred = round(preds[i][0], 2)
                img_name = f"{target}_{pred}_{img_name}"
                path = os.path.join(self.root, img_name)
                cv2.imwrite(path, visualization)
