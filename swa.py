import os
import glob

import torch
import numpy as np


def make_swa(path):
    out_file = os.path.join(path, "swa.pth") 
    iteration = glob.glob(os.path.join(path, "*pth"))
    to_remove = []
    for it in iteration:
        if "swa" in it:
            to_remove.append(it)

    for rem in to_remove:
        iteration.remove(rem)

    iteration = sorted(iteration, key=lambda x: float(x.split("_")[-1][:-4]), reverse=True)[:5]
    print(iteration)
    state_dict = None
    for mpath in iteration:
        f = torch.load(mpath, map_location=lambda storage, loc: storage)
        if state_dict is None:
            state_dict = f
        else:
            key = list(f.keys())
            for k in key:
                state_dict[k] = state_dict[k] + f[k]

    for k in key:
        state_dict[k] = state_dict[k] / len(iteration)
    print('')

    print(out_file)
    torch.save(state_dict, out_file)
    return iteration

def read_thresh(models, log_file):
    epochs = set()
    for model in models:
        name = os.path.split(model)[1]
        epoch = int(name.split("_")[-3])
        epochs.add(epoch)
    
    print("Epochs:", epochs)

    with open(log_file, "r") as fin:
        data = fin.read()
    
    epoch = 0
    threshs = []
    for line in data.split("\n"):
        if line[:3] == "Val":
            try:
                index = line.index("thresh_agg")
                thresh = float(line[index + 14:index + 20])
                if epoch in epochs:
                    threshs.append(thresh)
                    print(f"Epoch: {epoch} - thr: {thresh}")
                epoch += 1
            except:
                pass
    print("Mean:", np.mean(threshs))  

path = "outputs/tf_efficientnetv2_m/0_1/"

models = make_swa(path + "weights")
read_thresh(models, path + "log.txt")
