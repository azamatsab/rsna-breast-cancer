import yaml
from collections import OrderedDict

from attributedict.collections import AttributeDict


def get_configs(path):
    with open(path, "r") as fin:
        cfg = yaml.safe_load(fin)
    return AttributeDict(cfg)

def freeze(model, until):
    flag = False
    for name, param in model.named_parameters():
        if name == until:
            flag = True
        param.requires_grad = flag

def remove_parallel(old_state_dict):
    new_state_dict = OrderedDict()
    for k, v in old_state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    return new_state_dict