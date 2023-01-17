import os
import glob
import logging
import random
from collections import namedtuple

import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A

from .patch_dataset import RandomPatchDataset


class JointImgDataset(RandomPatchDataset):
    def __init__(self, dataframe, config, transform=None, train=True):
        super().__init__(dataframe, config, transform, train)
        patient_lat_dict = {pid: list() for pid in self.prediction_id}
        ImageInfo = namedtuple("ImageInfo", ["path", "target", "view", "pid"])
        for path, trg, pid, view in zip(self.image_paths, self.targets, self.prediction_id, self.views):
            patient_lat_dict[pid].append(ImageInfo(path, trg, view, pid))
        temp_lists = [patient_lat_dict[pid] for pid in patient_lat_dict]

        self.image_paths = [[img_info.path for img_info in imgs] for imgs in temp_lists]
        self.targets = [[img_info.target for img_info in imgs] for imgs in temp_lists]
        self.prediction_id = [[img_info.pid for img_info in imgs] for imgs in temp_lists]
        self.views = [[img_info.view for img_info in imgs] for imgs in temp_lists]

    def __getitem__(self, index):
        paths = self.image_paths[index]
        target = self.targets[index][0]
        pred_id = self.prediction_id[index][0]
        laterality = self.laterality[index][0]

        random.shuffle(paths)
        paths = paths[:2]
        imgs = []
        for path in paths:
            img = cv2.imread(os.path.join(self.img_path, path))
            if self.keep_ratio:
                img = pad(img, self.input_size)

            if self.train and target == 0 and np.random.uniform(0, 1) <= self.config.patch_prob:
                target = 1
                img = self.insert_patch(img, laterality)
            imgs.append(cv2.resize(img, (self.input_size[0] // 2, self.input_size[1])))
        img = np.concatenate(imgs, axis=1)

        if self.transform is not None:
            try:
                sample = self.transform(image=img)["image"]
            except Exception as err:
                logging.error(f"Error Occured: {err}, {path}")

        out = {"img": sample, "target": target, "pred_id": pred_id}
        if self.is_cam:
            out["bgr"] = cv2.resize(img, self.input_size)
            out["path"] = path
        return out
