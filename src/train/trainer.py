import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _inverse_scale(arr: np.ndarray, scaler) -> np.ndarray:
    flat = arr.reshape(-1, 1)
    inv = scaler.inverse_transform(flat)
    return inv.reshape(arr.shape)


# def _compute_metrics(
#     y_true: np.ndarray,
#     y_pred: np.ndarray,
#     scaler=None,
#     mape_epsilon: float = 1e-6,
# ) -> Dict[str, float]:
#     y_true = y_true.reshape(-1)
#     y_pred = y_pred.reshape(-1)

#     if scaler is not None:
#         y_true = _inverse_scale(y_true, scaler).reshape(-1)
#         y_pred = _inverse_scale(y_pred, scaler).reshape(-1)
        
#     diff = y_pred - y_true
#     mse = float(np.mean(diff ** 2))
#     rmse = float(np.sqrt(mse))
#     mae = float(np.mean(np.abs(diff)))
    
#     denom = np.abs(y_true)
#     # 对于光伏功率，如果实际值非常小（如夜间或早晚），会直接导致 MAPE 爆表
#     # 因此过滤掉低于最大装机量（或全局最大值）3% 的时间点来计算真实的 MAPE
#     capacity = np.max(y_true) if len(y_true) > 0 else 1.0
#     valid = denom > max(mape_epsilon, 0.03 * capacity)
    
#     if np.any(valid):
#         mape = float(np.mean(np.abs(diff[valid]) / denom[valid]) * 100.0)
#     else:
#         mape = 0.0
#     ss_res = float(np.sum(diff ** 2))
#     ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
#     r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
#     return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scaler=None,
    mape_epsilon: float = 1e-6,
    capacities: np.ndarray = None,
) -> Dict[str, float]:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    diff = y_pred - y_true
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    
    # 【修复核心】：把 R2 的计算移到外面，进行全天候全局计算！
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # 获取当前批次的最大装机量作为基准
    global_capacity = np.max(capacities) if capacities is not None and len(capacities) > 0 else (np.max(y_true) if len(y_true) > 0 else 1.0)
    
    # === 仅针对白天真实出力具有物理意义（功率大于总装机量 3%）阶段计算 MAPE ===
    daytime_mask = y_true > (0.03 * global_capacity)
    
    if np.any(daytime_mask) and np.sum(daytime_mask) > 1:
        y_true_day = y_true[daytime_mask]
        y_pred_day = y_pred[daytime_mask]
        
        # 使用 wMAPE 计算，更加抗造
        sum_true = np.sum(np.abs(y_true_day))
        mape = float(np.sum(np.abs(y_pred_day - y_true_day)) / sum_true * 100.0) if sum_true > 1e-6 else 0.0
    else:
        mape = 0.0

    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}

def run_one_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    epoch: Optional[int] = None,
    total_epochs: Optional[int] = None,
    desc: str = "train",
    show_progress: bool = True,
    compute_metrics: bool = False,
    scaler_y=None,
    mape_epsilon: float = 1e-6,
) -> Tuple[float, Dict[str, float]]:
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss = 0.0
    preds = []
    targets = []
    caps_list = []
    theory_list = []

    with torch.set_grad_enabled(train_mode):
        progress = loader
        if show_progress:
            epoch_text = f"{epoch + 1}/{total_epochs}" if epoch is not None and total_epochs is not None else ""
            progress = tqdm(loader, desc=f"{desc} {epoch_text}".strip(), leave=False)

        for batch_data in progress:
            # 兼容 dataset 返回两个（x,y）、三个（x,y,cap）或四个（x,y,cap,theory）元素
            if len(batch_data) == 4:
                batch_x, batch_y, batch_cap, batch_theory = batch_data
            elif len(batch_data) == 3:
                batch_x, batch_y, batch_cap = batch_data
                batch_theory = None
            else:
                batch_x, batch_y = batch_data
                batch_cap = None
                batch_theory = None
                
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pred = model(batch_x)
            loss = criterion(pred, batch_y)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                
                # === 优化 1：引入梯度裁剪，防止气象突变导致梯度爆炸跳出最优谷底 ===
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            if compute_metrics:
                preds.append(pred.detach().cpu())
                targets.append(batch_y.detach().cpu())
                if batch_cap is not None:
                    caps_list.append(batch_cap.cpu())
                if batch_theory is not None:
                    theory_list.append(batch_theory.cpu())

            if show_progress:
                progress.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader.dataset)
    metrics: Dict[str, float] = {}
    if compute_metrics and preds:
        y_pred = torch.cat(preds, dim=0).numpy()
        y_true = torch.cat(targets, dim=0).numpy()
        caps = torch.cat(caps_list, dim=0).numpy() if caps_list else None
        theory = torch.cat(theory_list, dim=0).numpy() if theory_list else None
        
        # 1. 残差的反归一化 (还原为物理标幺值残差)
        if scaler_y is not None:
            y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)
            y_true = scaler_y.inverse_transform(y_true.reshape(-1, 1)).reshape(y_true.shape)
            
        # 2. 从残差恢复到真实标幺值功率（真实功率 = 理论功率 - 残差）
        if theory is not None:
            if theory.size == y_pred.size:
                theory = theory.reshape(y_pred.shape)
            elif theory.ndim == 1:
                theory = theory.reshape(-1, 1)
            y_pred = theory - y_pred
            y_true = theory - y_true
            
        # 3. 乘以装机容量，恢复为绝对兆瓦物理功率 MW
        if caps is not None:
            if caps.size == y_pred.size:
                caps = caps.reshape(y_pred.shape)
            elif caps.ndim == 1:
                caps = caps.reshape(-1, 1)
            y_pred = y_pred * caps
            y_true = y_true * caps
        
        # 4. 计算指标（传入的已经是真实的绝对物理功率）
        metrics = _compute_metrics(y_true, y_pred, scaler=None, mape_epsilon=mape_epsilon, capacities=caps)

    return avg_loss, metrics


def train_model(model, train_loader, val_loader, cfg, device, scaler_y=None):
    # 更换鲁棒性更强的 HuberLoss，对异常的瞬时突变点（如光伏早晚波动）有更好的抗拉扯能力
    criterion = nn.HuberLoss(delta=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)

    # === 优化 2：引入学习率动态衰减调度器 ReduceLROnPlateau ===
    # 当验证集 loss 连续 3 个 epoch 停止下降时，将学习率减半，协助模型在极小值深坑处微调收敛
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3, 
        min_lr=1e-6
    )

    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / cfg.best_model_name
    last_path = model_dir / cfg.last_model_name

    history = []
    best_val = float("inf")
    patience = 0

    for epoch in range(cfg.num_epochs):
        train_loss, _ = run_one_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            epoch=epoch,
            total_epochs=cfg.num_epochs,
            desc="Train",
            show_progress=True,
            compute_metrics=False,
        )
        val_loss, val_metrics = run_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            epoch=epoch,
            total_epochs=cfg.num_epochs,
            desc="Val",
            show_progress=True,
            compute_metrics=True,
            scaler_y=scaler_y,
            mape_epsilon=cfg.mape_epsilon,
        )

        history.append({"train_loss": train_loss, "val_loss": val_loss, **val_metrics})
        metrics_text = (
            f"RMSE: {val_metrics.get('rmse', 0):.4f} | "
            f"MAE: {val_metrics.get('mae', 0):.4f} | "
            f"MAPE: {val_metrics.get('mape', 0):.2f}% | "
            f"R2: {val_metrics.get('r2', 0):.4f}"
        )
        print(
            f"Epoch [{epoch + 1}/{cfg.num_epochs}] "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | {metrics_text}"
        )

        # === 触发学习率衰减判决（传入当前的 val_loss） ===
        scheduler.step(val_loss)

        if val_loss < best_val - cfg.early_stopping_min_delta:
            best_val = val_loss
            patience = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience += 1
            if patience >= cfg.early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch + 1}, best val loss {best_val:.6f}."
                )
                break

    torch.save(model.state_dict(), last_path)

    return history
