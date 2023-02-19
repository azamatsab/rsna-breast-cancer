import os
import glob
import logging
import random

import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A


class BreastCancer(torch.utils.data.Dataset):
    def __init__(
        self,
        dataframe,
        config,
        transform=None,
        train=True,
    ):
        self.train = train
        self.input_size = config.img_size
        self.transform = transform
        self.is_cam = config.cam
        self.fda = config.fda

        if train:
            logging.info("Train Dataset")
        else:
            logging.info("Val Dataset")

        if "site_id" not in config:
            logging.info("##### ATTENTION: USING FULL DATASET WITHOUT SPLITTING BY SITE ID")
        else:
            dataframe = dataframe[dataframe.site_id == config.site_id]
            logging.info(f"##### ATTENTION: USING SITE ID {config.site_id}")

        # dataframe = dataframe[dataframe.cancer == 0]

        dataframe = self.balancing(dataframe, config, train)
        image_id = dataframe.image_id.tolist()
        patient_id = dataframe.patient_id.tolist()
        laterality = dataframe.laterality.tolist()

        self.image_paths = [f"{pid}/{iid}.png" for pid, iid in zip(patient_id, image_id)]

        image_paths_ = [f"{pid}_{iid}.png" for pid, iid in zip(patient_id, image_id)]
        self.site_ids = dataframe.site_id.tolist()

        self.siteid1 = []
        self.siteid2 = []
        for sid, imp in zip(self.site_ids, image_paths_):
            if sid == 1:
                self.siteid1.append(imp)
            else:
                self.siteid2.append(imp)
        
        self.targets = dataframe.cancer.tolist()
        self.views = dataframe.view.tolist()
        self.prediction_id = [f"{pid}_{lat}.png" for pid, lat in zip(patient_id, laterality)]
        self.laterality = laterality
        self.img_path = os.path.join(config.root_path, config.ds_path)
        self.keep_ratio = config.keep_ratio

        self.config = config

        logging.info(f"Disease: {self.targets.count(1)}; Safe: {self.targets.count(0)}")
        mids = dataframe.machine_id.tolist()
        for mid in set(mids):
            logging.info(f"Machine id: {mid}: {mids.count(mid)}")

    def balancing(self, dataframe, config, train):
        if train and config.upsample:
            df_1 = dataframe[dataframe.cancer == 1]
            df_0 = dataframe[dataframe.cancer == 0]
            df_list = []
            for i in range(config.upsample):
                df_list.append(df_1)
                new_pid = map(lambda x: str(x) + "_" + str(i), df_1.laterality.tolist())
                df_1["laterality"] = list(new_pid)
            dataframe = pd.concat(df_list + [df_0])
        if train and config.balance:
            df_1 = dataframe[dataframe.cancer == 1]
            df_0 = dataframe[dataframe.cancer == 0]
            df_0 = df_0.sample(len(df_1) * config.balance)
            dataframe = pd.concat([df_0, df_1])
        return dataframe

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        target = self.targets[index]
        pred_id = self.prediction_id[index]
        img = cv2.imread(os.path.join(self.img_path, path))
        #img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 3, 3)
        if self.keep_ratio:
            img = pad(img, self.input_size)

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
