import argparse
import os
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from config import ExperimentConfig, update_config
from data import build_distill_loader
from models import ResNet34Encoder
from losses import FeatureQueue, similarity_distribution, kl_distribution, statistic_loss
from utils import seed_everything, make_dir, save_checkpoint, load_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset")
    p.add_argument("--data_root")
    p.add_argument("--output_dir")
    p.add_argument("--seed", type=int)
    p.add_argument("--device")
    p.add_argument("--teacher_path", required=True)
    p.add_argument("--student_path", required=True)
    p.add_argument("--distill_epochs", type=int)
    p.add_argument("--distill_batch_size", type=int)
    p.add_argument("--distill_lr", type=float)
    return p.parse_args()


@torch.no_grad()
def warmup_queue(teacher, loader, queue, device, max_steps=32):
    teacher.eval()
    for i, (x, _) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        queue.enqueue(teacher(x))
        if queue.features.size(0) >= queue.size or i + 1 >= max_steps:
            break


def main():
    config = update_config(ExperimentConfig(), parse_args())
    seed_everything(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    make_dir(config.output_dir)
    loader = build_distill_loader(config)
    teacher = ResNet34Encoder(config.projection_dim, config.projection_hidden_dim).to(device)
    student = ResNet34Encoder(config.projection_dim, config.projection_hidden_dim).to(device)
    teacher.load_state_dict(load_checkpoint(config.teacher_path, map_location=device)["model"])
    student.load_state_dict(load_checkpoint(config.student_path, map_location=device)["model"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    optimizer = SGD(student.parameters(), lr=config.distill_lr, momentum=config.distill_momentum, weight_decay=config.distill_weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.distill_epochs)
    queue = FeatureQueue(config.projection_dim, config.queue_size, device)
    warmup_queue(teacher, loader, queue, device)
    for epoch in range(config.distill_epochs):
        student.train()
        total = 0.0
        count = 0
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            refs = queue.get()
            with torch.no_grad():
                zt = teacher(x)
                pt = similarity_distribution(zt, refs, config.teacher_temperature)
            zs = student(x)
            ps = similarity_distribution(zs, refs, config.student_temperature)
            loss = kl_distribution(pt, ps) + config.lambda_stat * statistic_loss(zt, zs)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            queue.enqueue(zt)
            total += loss.item() * x.size(0)
            count += x.size(0)
        scheduler.step()
        print({"epoch": epoch + 1, "loss": total / max(1, count)})
        save_checkpoint(os.path.join(config.output_dir, "defended_student_last.pt"), model=student.state_dict(), config=vars(config))
    save_checkpoint(os.path.join(config.output_dir, "defended_student_final.pt"), model=student.state_dict(), config=vars(config))


if __name__ == "__main__":
    main()
