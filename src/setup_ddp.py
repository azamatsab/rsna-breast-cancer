import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    if sys.platform == "win32":
        # Distributed package only covers collective communications with Gloo
        # backend and FileStore on Windows platform. Set init_method parameter
        # in init_process_group to a local file.
        # Example init_method="file:///f:/libtmp/some_file"
        init_method = "file:///{your local file path}"

        # initialize the process group
        dist.init_process_group(
            "gloo", init_method=init_method, rank=rank, world_size=world_size
        )
    else:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        # initialize the process group
        dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()
