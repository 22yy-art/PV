import sys
import os
from pathlib import Path

# 获取项目根目录并加入 sys.path，防止运行报错
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import numpy as np
from src.data.dataset import build_dataloaders
from src.models.lstm_model import PVLSTM
from src.config.settings import TrainConfig
from src.train.trainer import _compute_metrics

def evaluate_lstm():
    config = TrainConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. 获取数据和归一化器
    data_bundle, scaler_x, scaler_y = build_dataloaders(
        config.data_path,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        train_ratio=config.train_ratio,
        batch_size=config.batch_size
    )
    val_loader = data_bundle.val_loader
    
    # 2. 加载最佳 LSTM 模型
    best_lstm_path = os.path.join(config.model_dir, 'best_model.pt')
    lstm_model = PVLSTM(
        input_size=data_bundle.input_size, 
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        output_size=config.pred_len,
        num_stations=data_bundle.num_stations
    ).to(device)
    
    lstm_model.load_state_dict(torch.load(best_lstm_path, map_location=device))
    lstm_model.eval()
    
    lstm_preds_list = []
    y_true_list = []
    cap_list = []
    theory_list = []
    
    # 3. 进行推理
    print("开始使用 LSTM 进行验证集预测...")
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 4:
                x, y, cap, theory = batch
                cap_list.append(cap.cpu().numpy())
                theory_list.append(theory.cpu().numpy())
            elif len(batch) == 3:
                x, y, cap = batch
                cap_list.append(cap.cpu().numpy())
            else:
                x, y = batch
                
            x = x.to(device)
            lstm_preds = lstm_model(x)
            
            lstm_preds_list.append(lstm_preds.cpu().numpy())
            y_true_list.append(y.cpu().numpy())
            
    lstm_preds_arr = np.concatenate(lstm_preds_list, axis=0)
    y_true_arr = np.concatenate(y_true_list, axis=0)
    caps_arr = np.concatenate(cap_list, axis=0) if len(cap_list) > 0 else np.ones((len(y_true_arr),))
    theory_arr = np.concatenate(theory_list, axis=0) if len(theory_list) > 0 else None
    
    # 4. 反归一化并计算真实物理功率 (MW)
    # 因为现在的预测目标是残差(residual)，所以 inverse_transform 出来的是标幺值的残差
    lstm_preds_unscaled = scaler_y.inverse_transform(lstm_preds_arr.reshape(-1, 1)).reshape(lstm_preds_arr.shape)
    y_true_unscaled = scaler_y.inverse_transform(y_true_arr.reshape(-1, 1)).reshape(y_true_arr.shape)
    
    # 将残差还原回实际功率：实际功率 = 理论功率 - 残差
    if theory_arr is not None:
        if theory_arr.size == lstm_preds_unscaled.size:
            theory_aligned = theory_arr.reshape(lstm_preds_unscaled.shape)
            lstm_preds_unscaled = theory_aligned - lstm_preds_unscaled
            y_true_unscaled = theory_aligned - y_true_unscaled
        else:
            theory_aligned = theory_arr.reshape(-1, 1)
            lstm_preds_unscaled = theory_aligned - lstm_preds_unscaled
            y_true_unscaled = theory_aligned - y_true_unscaled
    
    # 最后乘以装机容量恢复为兆瓦(MW)
    if caps_arr.ndim > 1 and caps_arr.shape[1] == lstm_preds_arr.shape[1]:
        lstm_preds_unscaled = lstm_preds_unscaled * caps_arr
        y_true_unscaled = y_true_unscaled * caps_arr
    else:
        caps_arr = caps_arr.reshape(lstm_preds_arr.shape) if caps_arr.size == lstm_preds_arr.size else caps_arr.reshape(-1, 1)
        lstm_preds_unscaled = lstm_preds_unscaled * caps_arr
        y_true_unscaled = y_true_unscaled * caps_arr
        
    # 5. 计算指标
    metrics = _compute_metrics(y_true_unscaled, lstm_preds_unscaled, scaler=None, capacities=None)
    
    print("\n=== 最终单体 LSTM 模型评估结果 (物理功率 MW) ===")
    print(f"R2:   {metrics['r2']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"MAPE: {metrics['mape']:.2f}%")

if __name__ == '__main__':
    evaluate_lstm()