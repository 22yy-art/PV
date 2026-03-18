from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset, DataLoader

from .preprocess import load_and_clean_dataframe, FEATURE_COLS, TARGET_COL


@dataclass
class PVDataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    input_size: int
    num_stations: int  # 新增: 记录场站的总数量


class PVDataset(Dataset):
    def __init__(self, x_arr: np.ndarray, y_arr: np.ndarray, seq_len: int, pred_len: int):
        self.x_arr = x_arr
        self.y_arr = y_arr
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.x_arr) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index: int):
        x = self.x_arr[index:index + self.seq_len]
        y = self.y_arr[index + self.seq_len:index + self.seq_len + self.pred_len]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).squeeze(-1),
        )


def _build_site_samples(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    seq_len: int,
    pred_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    n = len(x_arr)
    max_start = n - seq_len - pred_len + 1
    if max_start <= 0:
        return np.empty((0, seq_len, x_arr.shape[1])), np.empty((0, pred_len, 1))

    for i in range(max_start):
        xs.append(x_arr[i:i + seq_len])
        ys.append(y_arr[i + seq_len:i + seq_len + pred_len])

    return np.stack(xs, axis=0), np.stack(ys, axis=0)


class PVWindowDataset(Dataset):
    def __init__(self, x_windows: np.ndarray, y_windows: np.ndarray, cap_windows: np.ndarray = None, theory_windows: np.ndarray = None):
        self.x_windows = x_windows
        self.y_windows = y_windows
        self.cap_windows = cap_windows
        self.theory_windows = theory_windows

    def __len__(self):
        return len(self.x_windows)

    def __getitem__(self, index: int):
        x = self.x_windows[index]
        y = self.y_windows[index]
        
        res = [
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).squeeze(-1)
        ]
        
        if self.cap_windows is not None:
            res.append(torch.tensor(self.cap_windows[index], dtype=torch.float32).squeeze(-1))
            
        if self.theory_windows is not None:
            res.append(torch.tensor(self.theory_windows[index], dtype=torch.float32).squeeze(-1))
            
        return tuple(res)


def build_dataloaders(
    data_path: str,
    seq_len: int,
    pred_len: int,
    train_ratio: float,
    batch_size: int,
    site_col: str = "location_new",
    time_col: str = "data_time",
    expected_freq: str = "15min",
    night_sr_threshold: float = 1e-6,
    num_workers: int = 0,
) -> Tuple[PVDataBundle, MinMaxScaler, MinMaxScaler]:
    df = load_and_clean_dataframe(
        data_path,
        site_col=site_col,
        time_col=time_col,
        expected_freq=expected_freq,
        night_sr_threshold=night_sr_threshold,
    )

    train_x_parts = []
    train_y_parts = []
    train_cap_parts = []
    train_site_parts = [] # 新增: 存储场站ID
    train_theory_parts = [] # 新增: 存储理论功率
    
    val_x_parts = []
    val_y_parts = []
    val_cap_parts = []
    val_site_parts = [] # 新增: 存储场站ID
    val_theory_parts = [] # 新增: 存储理论功率

    # 新增: 构建场站ID的编码映射 (将其转化为 0~N-1 的整数，供 Embedding 使用)
    unique_sites = sorted(df[site_col].unique())
    num_stations = len(unique_sites)
    site_to_id = {site: idx for idx, site in enumerate(unique_sites)}

    for _, site_df in df.groupby(site_col):
        site_df = site_df.sort_values(time_col).reset_index(drop=True)
        split_idx = int(len(site_df) * train_ratio)

        # 保障验证集可生成窗口
        if split_idx < seq_len + pred_len or (len(site_df) - split_idx) < seq_len + pred_len:
            continue

        site_id = site_to_id[site_df[site_col].iloc[0]]
        site_encoded = np.full((len(site_df), 1), site_id, dtype=np.float32)

        train_x_parts.append(site_df[FEATURE_COLS].values[:split_idx])
        train_y_parts.append(site_df[[TARGET_COL]].values[:split_idx])
        train_cap_parts.append(site_df[["install_power"]].values[:split_idx])
        train_site_parts.append(site_encoded[:split_idx])
        train_theory_parts.append(site_df[["theory_power"]].values[:split_idx])
        
        val_x_parts.append(site_df[FEATURE_COLS].values[split_idx:])
        val_y_parts.append(site_df[[TARGET_COL]].values[split_idx:])
        val_cap_parts.append(site_df[["install_power"]].values[split_idx:])
        val_site_parts.append(site_encoded[split_idx:])
        val_theory_parts.append(site_df[["theory_power"]].values[split_idx:])

    if not train_x_parts or not val_x_parts:
        raise ValueError("可用于构建窗口的数据不足，请检查数据长度、train_ratio、seq_len、pred_len")

    x_train_raw_all = np.concatenate(train_x_parts, axis=0)
    y_train_raw_all = np.concatenate(train_y_parts, axis=0)

    scaler_x = StandardScaler()
    scaler_y = MinMaxScaler()

    scaler_x.fit(x_train_raw_all)
    scaler_y.fit(y_train_raw_all)

    train_windows_x = []
    train_windows_y = []
    train_windows_cap = []
    train_windows_theory = []
    
    val_windows_x = []
    val_windows_y = []
    val_windows_cap = []
    val_windows_theory = []

    for tx, ty, tcap, tx_site, ttheory, vx, vy, vcap, vx_site, vtheory in zip(
        train_x_parts, train_y_parts, train_cap_parts, train_site_parts, train_theory_parts,
        val_x_parts, val_y_parts, val_cap_parts, val_site_parts, val_theory_parts
    ):
        tx_scaled = scaler_x.transform(tx)
        # 将未进行标准化的整数场站ID拼接到特征矩阵的最后一列
        tx_scaled = np.concatenate([tx_scaled, tx_site], axis=1)
        
        ty_scaled = scaler_y.transform(ty)
        
        vx_scaled = scaler_x.transform(vx)
        vx_scaled = np.concatenate([vx_scaled, vx_site], axis=1)
        
        vy_scaled = scaler_y.transform(vy)

        tx_w, ty_w = _build_site_samples(tx_scaled, ty_scaled, seq_len, pred_len)
        _, tcap_w = _build_site_samples(tx_scaled, tcap, seq_len, pred_len)  # 获取对应的容量窗口
        _, ttheory_w = _build_site_samples(tx_scaled, ttheory, seq_len, pred_len) # 理论功率窗口
        
        vx_w, vy_w = _build_site_samples(vx_scaled, vy_scaled, seq_len, pred_len)
        _, vcap_w = _build_site_samples(vx_scaled, vcap, seq_len, pred_len)
        _, vtheory_w = _build_site_samples(vx_scaled, vtheory, seq_len, pred_len)

        if len(tx_w) > 0:
            train_windows_x.append(tx_w)
            train_windows_y.append(ty_w)
            train_windows_cap.append(tcap_w)
            train_windows_theory.append(ttheory_w)
        if len(vx_w) > 0:
            val_windows_x.append(vx_w)
            val_windows_y.append(vy_w)
            val_windows_cap.append(vcap_w)
            val_windows_theory.append(vtheory_w)

    if not train_windows_x or not val_windows_x:
        raise ValueError("窗口构建失败，请检查数据是否存在连续时间段")

    x_train = np.concatenate(train_windows_x, axis=0)
    y_train = np.concatenate(train_windows_y, axis=0)
    cap_train = np.concatenate(train_windows_cap, axis=0)
    theory_train = np.concatenate(train_windows_theory, axis=0)
    
    x_val = np.concatenate(val_windows_x, axis=0)
    y_val = np.concatenate(val_windows_y, axis=0)
    cap_val = np.concatenate(val_windows_cap, axis=0)
    theory_val = np.concatenate(val_windows_theory, axis=0)

    train_ds = PVWindowDataset(x_train, y_train, cap_train, theory_train)
    val_ds = PVWindowDataset(x_val, y_val, cap_val, theory_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # x_train 的最后一列现在是整数ID了，但这不会影响 feature_dim，因为它不算气象特征
    feature_dim = len(FEATURE_COLS)
    
    bundle = PVDataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        input_size=feature_dim,
        num_stations=num_stations,
    )

    return bundle, scaler_x, scaler_y
