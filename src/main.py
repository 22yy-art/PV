import torch
from config.settings import TrainConfig
from data.dataset import build_dataloaders
from models.lstm_model import PVLSTM
from train.trainer import set_seed, train_model


def main():
    cfg = TrainConfig()
    set_seed(cfg.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_bundle, _, _ = build_dataloaders(
        data_path=cfg.data_path,
        seq_len=cfg.seq_len,
        pred_len=cfg.pred_len,
        train_ratio=cfg.train_ratio,
        batch_size=cfg.batch_size,
        site_col=cfg.site_col,
        time_col=cfg.time_col,
        expected_freq=cfg.expected_freq,
        night_sr_threshold=cfg.night_sr_threshold,
        num_workers=cfg.num_workers,
    )

    model = PVLSTM(
        input_size=data_bundle.input_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        output_size=cfg.pred_len,
        num_stations=data_bundle.num_stations, # 新增: 传递场站数量供 Embedding使用
    ).to(device)

    train_model(
        model=model,
        train_loader=data_bundle.train_loader,
        val_loader=data_bundle.val_loader,
        cfg=cfg,
        device=device,
    )


if __name__ == "__main__":
    
    main()
