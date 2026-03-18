from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    data_path: str = str(Path(__file__).resolve().parents[2] / "data" / "pv_data.csv")
    seq_len: int = 96  # 修改：从 24 改为 96，让模型能看到昨天同一时刻的状态
    pred_len: int = 4
    train_ratio: float = 0.8
    # === 优化 3：增大 Batch Size (从 64 提升至 256) 让每次梯度更新更平稳 ===
    batch_size: int = 256
    learning_rate: float = 5e-4 # 修改：网络变宽后，学习率稍微调小一点，让它下降得更平滑
    num_epochs: int = 150   # 修改：给模型多一点时间收敛
    hidden_size: int = 128  # 修改：从 64 提升到 128 或 256
    num_layers: int = 2
    num_workers: int = 0
    random_seed: int = 42
    site_col: str = "location_new"
    time_col: str = "data_time"
    expected_freq: str = "15min"
    night_sr_threshold: float = 1e-6
    model_dir: str = str(Path(__file__).resolve().parents[2] / "outputs" / "models")
    best_model_name: str = "best_model.pt"
    last_model_name: str = "last_model.pt"
    early_stopping_patience: int = 20  # 修改：从 5 提升到 20
    early_stopping_min_delta: float = 1e-4
    mape_epsilon: float = 1e-3
