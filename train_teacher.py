import argparse
import os
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from config import ExperimentConfig, update_config
from data import build_pretrain_loader
from models import ResNet34Encoder
from losses import info_nce
from utils import seed_everything, make_dir, save_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset")
    p.add_argument("--data_root")
    p.add_argument("--output_dir")
    p.add_argument("--seed", type=int)
    p.add_argument("--device")
    p.add_argument("--teacher_epochs", type=int)
    p.add_argument("--teacher_batch_size", type=int)
    p.add_argument("--teacher_lr", type=float)
    return p.parse_args()


def main():
    config = update_config(ExperimentConfig(), parse_args())
    seed_everything(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    make_dir(config.output_dir)
    loader = build_pretrain_loader(config)
    model = ResNet34Encoder(config.projection_dim, config.projection_hidden_dim).to(device)
    optimizer = SGD(model.parameters(), lr=config.teacher_lr, momentum=config.teacher_momentum, weight_decay=config.teacher_weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.teacher_epochs)
    for epoch in range(config.teacher_epochs):
        model.train()
        total = 0.0
        count = 0
        for (x1, x2), _ in loader:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            z1 = model(x1)
            z2 = model(x2)
            loss = info_nce(z1, z2, config.info_nce_temperature)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * x1.size(0)
            count += x1.size(0)
        scheduler.step()
        print({"epoch": epoch + 1, "loss": total / max(1, count)})
        save_checkpoint(os.path.join(config.output_dir, "teacher_last.pt"), model=model.state_dict(), config=vars(config))
    save_checkpoint(os.path.join(config.output_dir, "teacher_final.pt"), model=model.state_dict(), config=vars(config))


if __name__ == "__main__":
    main()
