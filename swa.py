import os
import glob

import torch


def make_swa(path):
    out_file = os.path.join(path, "swa.pth") 
    iteration = glob.glob(os.path.join(path, "*pth"))
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


make_swa("outputs/tf_efficientnetv2_m/3_0/weights")