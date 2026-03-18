# PV LSTM Forecast Project

基于 `pv_data.csv` 的光伏短时功率预测项目（PyTorch），采用模块化设计：

- 输入：连续 24 个时间步
- 输出：未来 4 个时间步
- 多站点联合：按 `location_new` 分站点构造窗口并合并训练
- 缺失处理：按 15 分钟频率补齐时间索引并分站点线性插值
- 夜间处理：`sr` 近零时目标功率置 0，并对异常负功率截断
- 编码兼容：自动尝试 `utf-8/gbk/gb2312/latin1` 读取
- 归一化：对特征与目标分别使用 `MinMaxScaler`（基于训练集拟合）

## 目录结构

- `src/config/settings.py`：训练配置
- `src/data/preprocess.py`：数据清洗与夜间数据处理
- `src/data/dataset.py`：数据集与 DataLoader 构建
- `src/models/lstm_model.py`：经典 LSTM 模型
- `src/train/trainer.py`：训练与验证流程
- `src/main.py`：训练入口

## 快速开始

```powershell
pip install -r requirements.txt
python .\src\main.py
```

## 可调参数

在 `src/config/settings.py` 中可修改：

- `seq_len=24`
- `pred_len=4`
- `batch_size`
- `num_epochs`
- `hidden_size`
- `num_layers`
- `site_col` / `time_col`
- `expected_freq`（默认 `15min`）
- `night_sr_threshold`

## 说明

当前默认采用分站点时间顺序切分训练/验证集（每个站点前 80% 训练，后 20% 验证）。
如需保存模型、添加早停、或多场站分组训练，可在现有模块上直接扩展。
