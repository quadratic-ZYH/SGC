import argparse
import os
import subprocess
import sys
from config import ExperimentConfig, update_config
from utils import make_dir, save_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset")
    p.add_argument("--data_root")
    p.add_argument("--output_dir")
    p.add_argument("--teacher_path", required=True)
    p.add_argument("--ratios", nargs="+", type=float, default=[0.0, 0.01, 0.03, 0.05, 0.07, 0.1])
    p.add_argument("--seed", type=int)
    p.add_argument("--device")
    return p.parse_args()


def run(cmd):
    print(" ".join(cmd))
    subprocess.check_call(cmd)


def main():
    args = parse_args()
    config = update_config(ExperimentConfig(), args)
    make_dir(config.output_dir)
    records = []
    for ratio in args.ratios:
        ratio_dir = os.path.join(config.output_dir, f"poison_{ratio:.3f}")
        make_dir(ratio_dir)
        run([sys.executable, "train_sgc.py", "--dataset", config.dataset, "--data_root", config.data_root, "--output_dir", ratio_dir, "--teacher_path", config.teacher_path, "--poisoning_ratio", str(ratio), "--seed", str(config.seed), "--device", config.device])
        run([sys.executable, "evaluate.py", "--dataset", config.dataset, "--data_root", config.data_root, "--output_dir", ratio_dir, "--encoder_path", os.path.join(ratio_dir, "sgc_student_final.pt"), "--seed", str(config.seed), "--device", config.device])
        records.append({"poisoning_ratio": ratio, "dir": ratio_dir})
    save_json(records, os.path.join(config.output_dir, "sensitivity_runs.json"))


if __name__ == "__main__":
    main()
