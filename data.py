import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class TwoCropTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return self.transform(x), self.transform(x)


def simclr_transform(size):
    return transforms.Compose([
        transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
    ])


def eval_transform(size):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])


def train_transform(size):
    return transforms.Compose([
        transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
    ])


def dataset_size(name):
    if name.lower() == "stl10":
        return 96
    return 32


def build_dataset(name, root, train=True, transform=None, unlabeled=False, download=True):
    name = name.lower()
    if name == "cifar10":
        return datasets.CIFAR10(root=root, train=train, transform=transform, download=download)
    if name == "stl10":
        split = "unlabeled" if unlabeled else ("train" if train else "test")
        return datasets.STL10(root=root, split=split, transform=transform, download=download)
    raise ValueError(name)


def build_pretrain_loader(config):
    size = dataset_size(config.dataset)
    dataset = build_dataset(config.dataset, config.data_root, train=True, transform=TwoCropTransform(simclr_transform(size)))
    return DataLoader(dataset, batch_size=config.teacher_batch_size, shuffle=True, num_workers=config.workers, pin_memory=True, drop_last=True)


def build_distill_loader(config):
    size = dataset_size(config.dataset)
    dataset = build_dataset(config.dataset, config.data_root, train=True, transform=train_transform(size))
    return DataLoader(dataset, batch_size=config.distill_batch_size, shuffle=True, num_workers=config.workers, pin_memory=True, drop_last=True)


def build_linear_loaders(config, dataset_name=None):
    name = dataset_name or config.dataset
    size = dataset_size(name)
    train_set = build_dataset(name, config.data_root, train=True, transform=eval_transform(size))
    test_set = build_dataset(name, config.data_root, train=False, transform=eval_transform(size))
    train_loader = DataLoader(train_set, batch_size=config.linear_batch_size, shuffle=True, num_workers=config.workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=config.linear_batch_size, shuffle=False, num_workers=config.workers, pin_memory=True)
    return train_loader, test_loader


def build_target_reference_loader(config):
    size = dataset_size(config.dataset)
    dataset = build_dataset(config.dataset, config.data_root, train=True, transform=eval_transform(size))
    labels = torch.as_tensor(dataset.targets if hasattr(dataset, "targets") else dataset.labels)
    idx = torch.nonzero(labels == config.target_class, as_tuple=False).flatten()
    idx = idx[:config.target_reference_size].tolist()
    subset = Subset(dataset, idx)
    return DataLoader(subset, batch_size=config.distill_batch_size, shuffle=True, num_workers=config.workers, pin_memory=True, drop_last=True)
