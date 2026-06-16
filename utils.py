import os
import random
import json
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def make_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path):
    with open(path, w, encoding="utf-8") as f:
        json.dump(obj, f)


def save_checkpoint(path, **items):
    make_dir(os.path.dirname(path))
    torch.save(items, path)


def load_checkpoint(path, map_location="cpu"):
    return torch.load(path, map_location=map_location)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self):
        return self.total / max(1, self.count)


def cycle(loader):
    while True:
        for batch in loader:
            yield batch
