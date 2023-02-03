import os
import glob
import logging
import random

import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A

from .breast_cancer import BreastCancer
from .preprocess import pad


class RandomPatchDataset(BreastCancer):
    def __init__(self, dataframe, config, transform=None, train=True):
        super().__init__(dataframe, config, transform, train)
        if self.config.patch_prob > 0:
            logging.info("##### Training using patches")

        self.patches = self.read_patches(self.config.patch_path)
        self.neg_patches = self.read_patches(self.config.neg_patch_path)
        
    def read_patches(self, patch):
        patches = glob.glob(f"{patch}/*png")
        site_patches = [[], []]
        for patch in patches:
            name = os.path.split(patch)[1]
            if name in set(self.siteid1):
                site_patches[0].append(patch)
            elif name in set(self.siteid2):
                site_patches[1].append(patch)

        logging.info(f"Patch size for site id 1: {len(site_patches[0])}")
        logging.info(f"Patch size for site id 2: {len(site_patches[1])}")
        return site_patches

    # TODO: Center crop
    def insert_patch(self, img, patch_path, laterality):
        imh, imw = img.shape[:2]
        pad_h = imh // 6
        pad_x = imw // 3

        patch = cv2.imread(patch_path)
        if self.fda:
            patch = aug(image=patch)["image"]
        pth, ptw = patch.shape[:2]
        pth = int(pth * np.random.uniform(0.5, 1.75))
        ptw = int(ptw * np.random.uniform(0.5, 1.75))

        if pth >= imh - 2 * pad_h:
            pth = imh - 2 * pad_h - 1
        if ptw >= imw - pad_x:
            ptw = imw - pad_x - 1

        y = np.random.randint(pad_h, imh - pth - pad_h)
        if "R" in laterality:
            x = np.random.randint(pad_x, imw - ptw)
        else:
            x = np.random.randint(0, imw - ptw - pad_x)

        img[y: y + pth, x : x + ptw] = cv2.resize(patch, (ptw, pth))
        return img

    def insert_patches(self, img, laterality, site_id, patches):
        patches = [np.random.choice(patches[site_id - 1])]
        # patches = np.random.choice(patches[site_id - 1], size=np.random.randint(1, 2))
        if self.fda:
            aug = A.Compose([A.FDA([img], p=1, read_fn=lambda x: x)])
        for patch_path in patches:
            img = self.insert_patch(img, patch_path, laterality)
        return img

    def center_crop(self, img):
        scale = np.random.uniform(0.85, 1)
        imh, imw = img.shape[:2]
        new_imh, new_imw = int(scale * imh), int(scale * imw)
        x0 = (imw - new_imw) // 2
        y0 = (imh - new_imh) // 2
        return img[y0:y0 + new_imh, x0:x0 + new_imw]

    def __getitem__(self, index):
        path = self.image_paths[index]
        target = self.targets[index]
        pred_id = self.prediction_id[index]
        laterality = self.laterality[index]
        site_id = self.site_ids[index]

        img = cv2.imread(os.path.join(self.img_path, path))
        if self.keep_ratio:
            img = pad(img, self.input_size)

        if self.train and np.random.uniform(0, 1) <= self.config.crop_prob:
            img = self.center_crop(img)

        if self.train and target == 0:
            if np.random.uniform(0, 1) <= self.config.patch_prob:
                target = 1
                img = self.insert_patches(img, laterality, site_id, self.patches)
                cv2.imwrite(f"{np.random.uniform(0, 1)}_{site_id}.png", img)
            elif np.random.uniform(0, 1) <= self.config.neg_patch_prob:
                img = self.insert_patches(img, laterality, site_id, self.neg_patches)
                cv2.imwrite(f"{np.random.uniform(0, 1)}_{site_id}.png", img)

        if self.transform is not None:
            try:
                sample = self.transform(image=img)["image"]
            except Exception as err:
                logging.error(f"Error Occured: {err}, {path}")

        out = {"img": sample, "target": target, "pred_id": pred_id, "view": self.views[index]}
        if self.is_cam:
            out["bgr"] = cv2.resize(img, self.input_size)
            out["path"] = path
        return out


class PatchDataset(BreastCancer):
    def _split(self, img):
        w, h = 256, 256
        tiles = [img[:, x:x+w,y:y+h] for x in range(0, img.shape[1], w) for y in range(0, img.shape[2], h)]
        random.shuffle(tiles)
        tiles = torch.cat(tiles, dim=0)
        return tiles

    def __getitem__(self, index):
        path = self.image_paths[index]
        target = self.targets[index]
        pred_id = self.prediction_id[index]

        img = cv2.imread(os.path.join(self.img_path, path))
        if self.keep_ratio:
            img = pad(img, self.input_size)
        if self.transform is not None:
            try:
                sample = self.transform(image=img)["image"]
                sample = self._split(sample)
            except Exception as err:
                logging.error(f"Error Occured: {err}, {path}")

        out = {"img": sample, "target": target, "pred_id": pred_id}
        if self.is_cam:
            out["bgr"] = cv2.resize(img, self.input_size)
            out["path"] = path
        return out