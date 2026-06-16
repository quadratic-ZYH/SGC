import argparse
import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from config import ExperimentConfig, update_config
from data import build_linear_loaders
from models import ResNet34Encoder, LinearProbe
from losses import add_patch_trigger
from utils import seed_everything, make_dir, save_checkpoint, load_checkpoint, save_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset")
    p.add_argument("--data_root")
    p.add_argument("--output_dir")
    p.add_argument("--seed", type=int)
    p.add_argument("--device")
    p.add_argument("--encoder_path", required=True)
    p.add_argument("--linear_epochs", type=int)
    p.add_argument("--linear_batch_size", type=int)
    p.add_argument("--linear_lr", type=float)
    p.add_argument("--target_class", type=int)
    return p.parse_args()


def train_linear(model, train_loader, config, device):
    model.train()
    optimizer = Adam(model.classifier.parameters(), lr=config.linear_lr, weight_decay=config.linear_weight_decay)
    for epoch in range(config.linear_epochs):
        total = 0.0
        count = 0
        correct = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * x.size(0)
            count += x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
        print({"epoch": epoch + 1, "loss": total / max(1, count), "acc": correct / max(1, count)})


@torch.no_grad()
def evaluate_acc(model, loader, device):
    model.eval()
    correct = 0
    count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        correct += (logits.argmax(dim=1) == y).sum().item()
        count += x.size(0)
    return 100.0 * correct / max(1, count)


@torch.no_grad()
def evaluate_asr(model, loader, config, device):
    model.eval()
    success = 0
    count = 0
    for x, y in loader:
        keep = y != config.target_class
        if keep.sum().item() == 0:
            continue
        x = x[keep].to(device, non_blocking=True)
        x = add_patch_trigger(x, config.trigger_size, config.trigger_value)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu()
        success += (pred == config.target_class).sum().item()
        count += x.size(0)
    return 100.0 * success / max(1, count)


def main():
    config = update_config(ExperimentConfig(), parse_args())
    seed_everything(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    make_dir(config.output_dir)
    train_loader, test_loader = build_linear_loaders(config, config.dataset)
    encoder = ResNet34Encoder(config.projection_dim, config.projection_hidden_dim).to(device)
    ckpt = load_checkpoint(config.encoder_path, map_location=device)
    encoder.load_state_dict(ckpt["model"])
    model = LinearProbe(encoder, encoder.feature_dim, config.num_classes).to(device)
    train_linear(model, train_loader, config, device)
    acc = evaluate_acc(model, test_loader, device)
    asr = evaluate_asr(model, test_loader, config, device)
    results = {"dataset": config.dataset, "acc": acc, "asr": asr}
    print(results)
    save_json(results, os.path.join(config.output_dir, f"linear_eval_{config.dataset}.json"))
    save_checkpoint(os.path.join(config.output_dir, f"linear_probe_{config.dataset}.pt"), model=model.state_dict(), config=vars(config))


if __name__ == "__main__":
    main()
