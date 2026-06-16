import argparse
import os
import subprocess
import sys
from config import ExperimentConfig, update_config
from utils import make_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset")
    p.add_argument("--data_root")
    p.add_argument("--output_dir")
    p.add_argument("--seed", type=int)
    p.add_argument("--device")
    p.add_argument("--skip_teacher", action="store_true")
    p.add_argument("--teacher_path")
    return p.parse_args()


def run(cmd):
    print(" ".join(cmd))
    subprocess.check_call(cmd)


def main():
    args = parse_args()
    config = update_config(ExperimentConfig(), args)
    make_dir(config.output_dir)
    teacher_path = args.teacher_path or os.path.join(config.output_dir, "teacher_final.pt")
    if not args.skip_teacher:
        run([sys.executable, "train_teacher.py", "--dataset", config.dataset, "--data_root", config.data_root, "--output_dir", config.output_dir, "--seed", str(config.seed), "--device", config.device])
    run([sys.executable, "train_sgc.py", "--dataset", config.dataset, "--data_root", config.data_root, "--output_dir", config.output_dir, "--teacher_path", teacher_path, "--seed", str(config.seed), "--device", config.device])
    student_path = os.path.join(config.output_dir, "sgc_student_final.pt")
    run([sys.executable, "evaluate.py", "--dataset", config.dataset, "--data_root", config.data_root, "--output_dir", config.output_dir, "--encoder_path", student_path, "--seed", str(config.seed), "--device", config.device])
    run([sys.executable, "visualize_tsne.py", "--dataset", config.dataset, "--data_root", config.data_root, "--output_dir", config.output_dir, "--encoder_path", student_path, "--seed", str(config.seed), "--device", config.device])


if __name__ == "__main__":
    main()
