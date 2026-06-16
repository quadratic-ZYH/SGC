import argparse
import os
import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from config import ExperimentConfig, update_config
from data import build_distill_loader, build_target_reference_loader
from models import ResNet34Encoder
from losses import FeatureQueue, add_patch_trigger, similarity_distribution, kl_distribution, statistic_loss, flatten_grads, assign_flat_grad, collaborative_gradient
from utils import seed_everything, make_dir, save_checkpoint, load_checkpoint, cycle


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset")
    p.add_argument("--data_root")
    p.add_argument("--output_dir")
    p.add_argument("--seed", type=int)
    p.add_argument("--device")
    p.add_argument("--teacher_path", required=True)
    p.add_argument("--distill_epochs", type=int)
    p.add_argument("--distill_batch_size", type=int)
    p.add_argument("--distill_lr", type=float)
    p.add_argument("--poisoning_ratio", type=float)
    p.add_argument("--target_class", type=int)
    return p.parse_args()


@torch.no_grad()
def warmup_queue(teacher, loader, queue, device, max_steps=32):
    teacher.eval()
    for i, (x, _) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        queue.enqueue(teacher(x))
        if queue.features.size(0) >= queue.size or i + 1 >= max_steps:
            break


def benign_loss(teacher, student, x, refs, config):
    with torch.no_grad():
        zt = teacher(x)
        pt = similarity_distribution(zt, refs, config.teacher_temperature)
    zs = student(x)
    ps = similarity_distribution(zs, refs, config.student_temperature)
    ld = kl_distribution(pt, ps)
    ls = statistic_loss(zt, zs)
    return ld + config.lambda_stat * ls, zt


def backdoor_loss(teacher, student, x, target_x, refs, config):
    if x.size(0) == 0:
        return None
    xt = add_patch_trigger(x, config.trigger_size, config.trigger_value)
    with torch.no_grad():
        zref = teacher(target_x)
        q = similarity_distribution(zref, refs, config.teacher_temperature).mean(dim=0, keepdim=True)
        q = q.expand(xt.size(0), -1)
    zs = student(xt)
    ps = similarity_distribution(zs, refs, config.student_temperature)
    return config.lambda_bd * kl_distribution(q, ps)


def main():
    config = update_config(ExperimentConfig(), parse_args())
    seed_everything(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    make_dir(config.output_dir)
    loader = build_distill_loader(config)
    target_loader = cycle(build_target_reference_loader(config))
    teacher = ResNet34Encoder(config.projection_dim, config.projection_hidden_dim).to(device)
    student = ResNet34Encoder(config.projection_dim, config.projection_hidden_dim).to(device)
    ckpt = load_checkpoint(config.teacher_path, map_location=device)
    teacher.load_state_dict(ckpt["model"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    optimizer = SGD(student.parameters(), lr=config.distill_lr, momentum=config.distill_momentum, weight_decay=config.distill_weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.distill_epochs)
    params = [p for p in student.parameters() if p.requires_grad]
    queue = FeatureQueue(config.projection_dim, config.queue_size, device)
    warmup_queue(teacher, loader, queue, device)
    for epoch in range(config.distill_epochs):
        student.train()
        total_b = 0.0
        total_d = 0.0
        total_w = 0.0
        count = 0
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            refs = queue.get()
            n_poison = max(1, int(x.size(0) * config.poisoning_ratio))
            x_bd = x[:n_poison]
            target_x, _ = next(target_loader)
            target_x = target_x.to(device, non_blocking=True)
            lb, zt = benign_loss(teacher, student, x, refs, config)
            optimizer.zero_grad(set_to_none=True)
            lb.backward(retain_graph=True)
            gb = flatten_grads(params)
            optimizer.zero_grad(set_to_none=True)
            ld = backdoor_loss(teacher, student, x_bd, target_x, refs, config)
            if ld is None:
                ld = torch.zeros((), device=device)
            ld.backward()
            gd = flatten_grads(params)
            g, w = collaborative_gradient(gb, gd)
            optimizer.zero_grad(set_to_none=True)
            assign_flat_grad(params, g)
            optimizer.step()
            queue.enqueue(zt)
            total_b += lb.item() * x.size(0)
            total_d += ld.item() * x.size(0)
            total_w += w * x.size(0)
            count += x.size(0)
        scheduler.step()
        print({"epoch": epoch + 1, "benign_loss": total_b / max(1, count), "backdoor_loss": total_d / max(1, count), "w": total_w / max(1, count)})
        save_checkpoint(os.path.join(config.output_dir, "sgc_student_last.pt"), model=student.state_dict(), config=vars(config))
    save_checkpoint(os.path.join(config.output_dir, "sgc_student_final.pt"), model=student.state_dict(), config=vars(config))


if __name__ == "__main__":
    main()
