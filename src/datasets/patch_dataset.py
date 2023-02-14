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

        self.patch_prob = self.config.patch_prob

        if train:
            self.patches = self.read_patches(self.config.patch_path, extra=True)
            self.target_patches = self.read_patches(self.config.trg_patch_path, extra=False)

    def is_ddcm(self, path):
        if "/" in path:
            path = os.path.split(path)[1]
        return "img" in path

    def read_patches(self, patch, extra):
        patches = glob.glob(f"{patch}/*")
        site_patches = [[], []]
        for patch in patches:
            name = os.path.split(patch)[1]
            if extra and self.is_ddcm(name):
                site_patches[0].append(patch)
                site_patches[1].append(patch)
            else:
                if name in set(self.siteid1):
                    site_patches[0].append(patch)
                elif name in set(self.siteid2):
                    site_patches[1].append(patch)

        logging.info(f"Patch size for site id 1: {len(site_patches[0])}")
        logging.info(f"Patch size for site id 2: {len(site_patches[1])}")
        return site_patches

    def insert_patch(self, img, patch_path, laterality, aug):
        imh, imw = img.shape[:2]
        pad_h = imh // 6
        pad_x = imw // 3

        patch = cv2.imread(patch_path)
        flags = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
        degree = np.random.choice(flags)
        if degree is not None:
            patch = cv2.rotate(patch, degree)

        if aug is not None:
            patch = aug(image=patch)["image"]
        pth, ptw = patch.shape[:2]
        pth = int(pth * np.random.uniform(0.5, 1.75))
        ptw = int(ptw * np.random.uniform(0.5, 1.75))

        if pth >= imh - 2 * pad_h:
            pth = imh - 2 * pad_h - 1
        if ptw >= imw - pad_x:
            ptw = imw - pad_x - 1

        y = np.random.randint(pad_h, imh - pth - pad_h)
        if laterality == "R":
            x = np.random.randint(pad_x, imw - ptw)
        else:
            x = np.random.randint(0, imw - ptw - pad_x)

        img[y: y + pth, x : x + ptw] = cv2.resize(patch, (ptw, pth))
        return img

    def insert_patches(self, img, laterality, site_id, patches):
        # patches = [np.random.choice(patches[site_id - 1])]
        patches = np.random.choice(patches[site_id - 1], size=np.random.randint(1, 3))
        for patch_path in patches:
            aug = None
            if self.fda and self.is_ddcm(patch_path):
                trgt_pth = np.random.choice(self.target_patches[site_id - 1])
                trgt = cv2.imread(trgt_pth)
                aug = A.PixelDistributionAdaptation([trgt], blend_ratio=(1.0, 1.0), p=1, read_fn=lambda x: x)
            img = self.insert_patch(img, patch_path, laterality, aug)
        return img

    def center_crop(self, img):
        scale = np.random.uniform(0.85, 1)
        imh, imw = img.shape[:2]
        new_imh, new_imw = int(scale * imh), int(scale * imw)
        x0 = (imw - new_imw) // 2
        y0 = (imh - new_imh) // 2
        return img[y0:y0 + new_imh, x0:x0 + new_imw]

    def get(self, index, patch_prob):
        path = self.image_paths[index]
        target = self.targets[index]
        laterality = self.laterality[index]
        site_id = self.site_ids[index]

        img = cv2.imread(os.path.join(self.img_path, path))
        if self.keep_ratio:
            img = pad(img, self.input_size)

        if self.train and np.random.uniform(0, 1) < self.config.crop_prob:
            img = self.center_crop(img)

        if self.train and target == 0:
            if np.random.uniform(0, 1) < patch_prob:
                target = 1
                img = self.insert_patches(img, laterality, site_id, self.patches)
        return img, target

    def merge_hor(self, img0, img1):
        h = img0.shape[0]
        img0[h // 2: ] = img1[h // 2: ]
        return img0

    def merge_diagonal(self, img0, img1):
        combined = np.ones_like(img0) * 255
        angle = -45
        lower_intersection = 0.2

        y, x, _ = img1.shape

        yy, xx = np.mgrid[:y, :x]
        img0_positions = (xx - lower_intersection * x) * np.tan(angle) > (yy - y)
        img1_positions = (xx - lower_intersection * x) * np.tan(angle) < (yy - y)

        combined[img0_positions] = img0[img0_positions]
        combined[img1_positions] = img1[img1_positions]
        return combined

    def merge(self, img0, img1):
        h, w, _ = img0.shape
        img0 = cv2.resize(img0, (h, h))
        img1 = cv2.resize(img1, (h, h))
        if np.random.uniform(0, 1) < 0.5:
            img = self.merge_hor(img0, img1)
        else:
            img = self.merge_diagonal(img0, img1)
        img = cv2.resize(img, (w, h))
        return img

    def __getitem__(self, index):
        img, target = self.get(index, self.patch_prob)
        if self.train and np.random.uniform(0, 1) < 0.5:
            if target == 1:
                img_, target_ = self.get(np.random.randint(len(self.image_paths)), 2)
                assert target == target_, (target, target_)
                img = self.merge(img, img_)
            else:
                img_, target_ = self.get(np.random.randint(len(self.image_paths)), -1)
                if target_ == 0:
                    img = self.merge(img, img_)

        if self.transform is not None:
            try:
                sample = self.transform(image=img)["image"]
            except Exception as err:
                logging.error(f"Error Occured: {err}, {path}")

        out = {"img": sample, "target": target, "pred_id": self.prediction_id[index], "view": self.views[index]}
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