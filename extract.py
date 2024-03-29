import os
import glob

import cv2
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm


detector = torch.hub.load('../yolov5', 'custom', path='../yolov5/rsna-roi-003.pt', source='local')


def crop_coords(img):
    """
    Crop ROI from image.
    """
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, breast_mask = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    cnts, _ = cv2.findContours(breast_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key = cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return (x, y, w, h)


def crop(folders, new_root):
    size = 768
    counter = 0
    for folder in tqdm(folders):
        paths = glob.glob(os.path.join(folder, "*png"))
        folder_name = folder.split("/")[-1]
        for path in paths:
            frame = cv2.imread(path)
            filename = os.path.split(path)[-1]
            h, w = frame.shape[:2]
            hratio = h / size
            wratio = w / size
            detections = detector(cv2.resize(frame, (size, size)))
            results = detections.pandas().xyxy[0].to_dict(orient="records")
            new_path = os.path.join(new_root, folder_name)
            os.makedirs(new_path, exist_ok=True)
            new_path = os.path.join(new_path, filename)
            if len(results) > 0:
                result = results[0]
                x1, y1 = int(result["xmin"] * wratio), int(result["ymin"] * hratio)
                x2, y2 = int(result["xmax"] * wratio), int(result["ymax"] * hratio)
                cv2.imwrite(new_path, frame[y1:y2, x1:x2])
            else:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                x, y, w, h = crop_coords(img)
                cv2.imwrite(new_path, frame[y:y + h, x:x + w])
                print(new_path)
                counter += 1
    print(counter)


root = "ims_2048/data/train_images"
# csv_path = "/data/zhan/compets/rsna_breast/data/TheChineseMammographyDatabase/malignant.csv"
new_root = "ims_2048/data/roi_train_images"
os.makedirs(new_root, exist_ok=True)
folders = glob.glob(os.path.join(root, "*"))

crop(folders, new_root)
