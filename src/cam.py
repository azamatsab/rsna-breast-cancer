import os

from tqdm import tqdm
import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import cv2
import numpy as np


def gradcam2bbox(gradcam, thr=0.3):
    # Binarize the image
    mask = (gradcam > thr).astype("uint8")

    # Make contours around the binarized image, keep only the largest contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)

    # Find ROI from largest contour
    ys = contour.squeeze()[:, 0]
    xs = contour.squeeze()[:, 1]
    
    y1 = np.min(xs); y2 = np.max(xs)
    x1 = np.min(ys); x2 = np.max(ys)
    
    pad = 25

    x1 = max(x1 - pad, 0)
    x2 = min(x2 + pad, gradcam.shape[1])

    y1 = max(y1 - pad, 0)
    y2 = min(y2 + pad, gradcam.shape[0])
    return [x1, y1, x2, y2]

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

    def cam_to_image(self, img_path):
        test_tr = self.model.test_transform()
        img = cv2.imread(img_path)
        img = img[:-320, :-160]
        tensor_img = test_tr(image=img)["image"]
        data = {"img": tensor_img.unsqueeze(0), "target": torch.tensor([1])}
        with torch.no_grad():
            _, preds = self.model.iteration(data, False)
        pred = self.sigmoid(preds).detach().cpu().numpy()[0][0]
        print(pred)
        ctargets = [ClassifierOutputTarget(0)]
        cam_img = self.cam(input_tensor=tensor_img.unsqueeze(0), targets=ctargets)[0]

        img = cv2.resize(img, self.config.img_size)
        visualization = show_cam_on_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255, cam_img, use_rgb=True)
        x1, y1, x2, y2 = gradcam2bbox(cam_img)
        orig_name = "cam_box.png"
        cv2.rectangle(visualization, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.imwrite(orig_name, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

    def calculate_cam(self, loader):
        counter = 0
        for data in tqdm(loader):
            input_tensor = data["img"]
            targets = data["target"]
            ctargets = [ClassifierOutputTarget(0) for trg in targets]
            grayscale_cam = self.cam(input_tensor=input_tensor, targets=ctargets)
            with torch.no_grad():
                _, preds = self.model.iteration(data, False)
            preds = self.sigmoid(preds).detach().cpu().numpy()
            imgs = data["bgr"].numpy()
            for i, cam_img in enumerate(grayscale_cam):
                ithimg = imgs[i]
                ithimg = ithimg[:-200]
                cam_img = cam_img[:-200]
                try:
                    x1, y1, x2, y2 = gradcam2bbox(cam_img)
                except:
                    counter += 1
                    continue
                visualization = show_cam_on_image(cv2.cvtColor(ithimg, cv2.COLOR_BGR2RGB) / 255, cam_img, use_rgb=True)
                img_name = data["path"][i].replace("/", "_")
                folder = img_name[:-4]
                target = targets[i]
                pred = round(preds[i][0], 2)
                if pred > 0.65:
                    root = os.path.join(self.root, folder)
                    os.makedirs(root, exist_ok=True)
                    vis_name = f"{target}_{pred}_{img_name}"
                    path = os.path.join(root, vis_name)
                    cv2.rectangle(visualization, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    flag = cv2.imwrite(path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

                    patch_name = img_name
                    path = os.path.join(root, patch_name)
                    cv2.imwrite(path, ithimg[y1:y2, x1:x2])

                    orig_name = "orig.png"
                    path = os.path.join(root, orig_name)
                    cv2.imwrite(path, ithimg)

                    orig_name = "orig_box.png"
                    path = os.path.join(root, orig_name)
                    cv2.rectangle(ithimg, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.imwrite(path, ithimg)
        print(counter)