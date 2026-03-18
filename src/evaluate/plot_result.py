import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.dataset import build_dataloaders
from src.models.lstm_model import PVLSTM
from src.config.settings import TrainConfig

def plot_predictions():
    config = TrainConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    data_bundle, scaler_x, scaler_y = build_dataloaders(
        config.data_path,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        train_ratio=config.train_ratio,
        batch_size=config.batch_size
    )
    val_loader = data_bundle.val_loader
    
    model_path = os.path.join(config.model_dir, 'best_model.pt')
    model = PVLSTM(
        input_size=data_bundle.input_size, 
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        output_size=config.pred_len,
        num_stations=data_bundle.num_stations
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    preds_list, trues_list, caps_list, theory_list = [], [], [], []
    
    print('正在生成验证集预测结果...')
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 4:
                x, y, cap, theory = batch
            else:
                raise ValueError('DataLoader 需要返回4个元素(x, y, cap, theory)以支持残差还原！')
            
            x = x.to(device)
            pred = model(x)
            
            preds_list.append(pred.cpu().numpy())
            trues_list.append(y.cpu().numpy())
            caps_list.append(cap.cpu().numpy())
            theory_list.append(theory.cpu().numpy())
            
    preds_arr = np.concatenate(preds_list, axis=0)
    trues_arr = np.concatenate(trues_list, axis=0)
    caps_arr = np.concatenate(caps_list, axis=0)
    theory_arr = np.concatenate(theory_list, axis=0)
    
    preds_unscaled = scaler_y.inverse_transform(preds_arr.reshape(-1, 1)).reshape(preds_arr.shape)
    trues_unscaled = scaler_y.inverse_transform(trues_arr.reshape(-1, 1)).reshape(trues_arr.shape)
    
    theory_arr = theory_arr.reshape(preds_unscaled.shape)
    caps_arr = caps_arr.reshape(preds_unscaled.shape) if caps_arr.size == preds_unscaled.size else caps_arr.reshape(-1, 1)
    
    pred_power_mw = (theory_arr - preds_unscaled) * caps_arr
    true_power_mw = (theory_arr - trues_unscaled) * caps_arr
    
    pred_line = pred_power_mw[:, 0]
    true_line = true_power_mw[:, 0]
    
    start_idx = 400
    window_size = 96 * 3 
    if start_idx + window_size > len(true_line):
        start_idx = 0 
        
    plot_pred = pred_line[start_idx : start_idx + window_size]
    plot_true = true_line[start_idx : start_idx + window_size]
    
    print('正在绘制功率对比曲线...')
    plt.figure(figsize=(14, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    x_axis = np.arange(len(plot_true))
    
    plt.plot(x_axis, plot_true, label='实际物理功率 (Actual Power)', color='#1f77b4', linewidth=2.5, alpha=0.8)
    plt.plot(x_axis, plot_pred, label='LSTM 预测功率 (Predicted Power)', color='#d62728', linestyle='--', linewidth=2, alpha=0.9)
    
    plt.title('光伏集群绝对物理功率预测对比 (连续 3 天)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('时间序列 (15分钟/步)', fontsize=12)
    plt.ylabel('发电功率 (MW)', fontsize=12)
    
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.ylim(bottom=0)
    
    out_dir = Path(project_root) / 'outputs' / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'residual_prediction_curve.png'
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f'🎉 绘图成功！高清对比曲线已保存至: {out_path}')

if __name__ == '__main__':
    plot_predictions()
