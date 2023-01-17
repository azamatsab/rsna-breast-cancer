import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from src.trainer import Trainer
from src.setup_ddp import setup, cleanup


def run_train(rank, world_size):
    setup(rank, world_size)
    # create model and move it to GPU with id rank
    trainer = Trainer(model, config, rank)
    trainer.fit(train_dataset, val_dataset)
    cleanup()


def run(fn, world_size):
    mp.spawn(fn, args=(world_size,), nprocs=world_size, join=True)
