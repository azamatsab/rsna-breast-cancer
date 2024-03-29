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
            self.patches = self.read_patches(self.config.patch_path)

        self.patch_transform = A.Compose([
                                    A.OneOf([ 
                                            A.RandomContrast(),
                                            A.RandomGamma(),
                                            A.RandomBrightness(),
                                            ], p=0.95),
                                    A.OneOf([
                                            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                                            A.GridDistortion(),
                                            A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                                            ], p=0.95),
                                ], p=1
                                )

    def balancing(self, dataframe, config, train):
        image_id = dataframe.image_id.tolist()
        patient_id = dataframe.patient_id.tolist()
        image_paths_ = [f"{pid}_{iid}.png" for pid, iid in zip(patient_id, image_id)]
        dataframe["img_name"] = image_paths_
        if train and config.upsample:
            patches = glob.glob(f"{self.config.patch_path}/*")
            patches = set([os.path.split(patch)[1] for patch in patches])
            df_1 = dataframe[dataframe.cancer == 1]
            df_0 = dataframe[dataframe.cancer == 0]

            df1_easy = df_1[df_1["img_name"].isin(patches)]
            df1_hard = df_1[~df_1["img_name"].isin(patches)]
            logging.info(f"Easy examples: {len(df1_easy)}")
            logging.info(f"Hard examples: {len(df1_hard)}")

            df_list = []
            for i in range(config.upsample):
                df_list.append(df1_easy)
                new_pid = map(lambda x: str(x) + "_" + str(i), df1_easy.laterality.tolist())
                df1_easy["laterality"] = list(new_pid)
            for i in range(config.hard_upsample):
                df_list.append(df1_hard)
                new_pid = map(lambda x: str(x) + "_" + str(i), df1_hard.laterality.tolist())
                df1_hard["laterality"] = list(new_pid)

            dataframe = pd.concat(df_list + [df_0])
        if train and config.balance:
            df_1 = dataframe[dataframe.cancer == 1]
            df_0 = dataframe[dataframe.cancer == 0]
            df_0 = df_0.sample(len(df_1) * config.balance)
            dataframe = pd.concat([df_0, df_1])
        return dataframe

    def read_patches(self, patch):
        patches = glob.glob(f"{patch}/*")
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

    def insert_patch(self, img, patch, laterality, aug):
        imh, imw = img.shape[:2]
        pad_h = imh // 6
        pad_x = imw // 3

        flags = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
        degree = np.random.choice(flags)
        if degree is not None:
            patch = cv2.rotate(patch, degree)

        if aug is not None:
            patch = aug(image=patch)["image"]
        pth, ptw = patch.shape[:2]

        scale1 = np.random.uniform(0.5, 1.75)
        scale2 = np.random.uniform(0.5, 1.75)
        pth = min(256, int(pth * scale1))
        ptw = min(256, int(ptw * scale2))

        if pth >= imh - 2 * pad_h:
            pth = imh - 2 * pad_h - 1
        if ptw >= imw - pad_x:
            ptw = imw - pad_x - 1

        y = np.random.randint(pad_h, imh - pth - pad_h)
        if laterality == "R":
            x = np.random.randint(pad_x, imw - ptw)
        else:
            x = np.random.randint(0, imw - ptw - pad_x)

        patch = cv2.resize(patch, (ptw, pth))
        alpha_s = patch[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            img[y:y + pth, x:x + ptw, c] = (alpha_s * patch[:, :, c] +
                                    alpha_l * img[y:y + pth, x:x + ptw, c])

        return img

    def insert_patches(self, img, laterality, site_id, patches):
        if len(self.config.patch_num) == 2:
            low, high = self.config.patch_num
            single_patches = np.random.choice(patches[site_id - 1], size=np.random.randint(low, high))
        else:
            single_patches = [np.random.choice(patches[site_id - 1])]

        patch_imgs = [cv2.imread(path, -1) for path in single_patches]

        if len(self.config.double_patch_num) == 2:
            low, high = self.config.double_patch_num
            double_patches = np.random.choice(patches[site_id - 1], size=np.random.randint(low, high))
            for ind in range(0, len(double_patches) - 1, 2):
                patch1 = cv2.imread(double_patches[ind], -1)
                patch2 = cv2.imread(double_patches[ind + 1], -1)
                merged = self.merge(patch1, patch2)
                patch_imgs.append(merged)

        for patch in patch_imgs:
            aug = None
            if self.config.patch_aug:
                patch = self.patch_transform(image=patch)["image"]
            img = self.insert_patch(img, patch, laterality, aug)
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

    def __getitem__(self, index):
        img, target = self.get(index, self.patch_prob)

        if self.transform is not None:
            try:
                sample = self.transform(image=img)["image"]
            except Exception as err:
                logging.error(f"Error Occured: {err}")

        out = {"img": sample, "target": target, "pred_id": self.prediction_id[index], "view": self.views[index]}
        if self.is_cam:
            out["bgr"] = cv2.resize(img, self.input_size)
            out["path"] = path
        return out

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
        max_len = max(w, h)
        img0 = cv2.resize(img0, (max_len, max_len))
        img1 = cv2.resize(img1, (max_len, max_len))
        if np.random.uniform(0, 1) < 0.5:
            img = self.merge_hor(img0, img1)
        else:
            img = self.merge_diagonal(img0, img1)
        img = cv2.resize(img, (w, h))
        return img

    def unpad_patch(self, img):
        h, w, _ = img.shape
        pad = 20
        if h > 4 * pad:
            y1, y2 = pad, h - pad
        if w > 4 * pad:
            x1, x2 = pad, w - pad
        return img[y1:y2, x1:x2]


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