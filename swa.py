import os
import glob

import torch


def make_swa(path):
    out_file = os.path.join(path, "swa.pth") 
    iteration = glob.glob(os.path.join(path, "*pth"))
    iteration = [
            "outputs/tf_efficientnetv2_s/87_1/weights/tf_efficientnetv2_s_7_0.762_0.3437.pth",
            "outputs/tf_efficientnetv2_s/87_1/weights/tf_efficientnetv2_s_9_0.7598_0.35.pth",
            "outputs/tf_efficientnetv2_s/87_1/weights/tf_efficientnetv2_s_10_0.7691_0.35.pth",
            "outputs/tf_efficientnetv2_s/87_1/weights/tf_efficientnetv2_s_11_0.771_0.3478.pth"
        ]
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


make_swa("outputs/tf_efficientnetv2_s/87_1/weights")