from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    dataset: str = "cifar10"
    data_root: str = "./data"
    output_dir: str = "./outputs"
    seed: int = 0
    device: str = "cuda"
    workers: int = 4
    teacher_epochs: int = 800
    teacher_batch_size: int = 512
    teacher_lr: float = 0.5
    teacher_weight_decay: float = 1e-4
    teacher_momentum: float = 0.9
    distill_epochs: int = 200
    distill_batch_size: int = 256
    distill_lr: float = 1e-3
    distill_weight_decay: float = 5e-4
    distill_momentum: float = 0.9
    linear_epochs: int = 100
    linear_batch_size: int = 256
    linear_lr: float = 1e-3
    linear_weight_decay: float = 1e-6
    projection_dim: int = 128
    projection_hidden_dim: int = 2048
    teacher_temperature: float = 0.2
    student_temperature: float = 0.2
    info_nce_temperature: float = 0.2
    lambda_stat: float = 1.0
    lambda_bd: float = 1.0
    queue_size: int = 4096
    poisoning_ratio: float = 0.05
    target_class: int = 0
    target_reference_size: int = 500
    trigger_size: int = 4
    trigger_value: float = 1.0
    num_classes: int = 10


def update_config(config, args):
    for key, value in vars(args).items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)
    return config
