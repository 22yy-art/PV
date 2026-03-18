import pandas as pd
import numpy as np
FEATURE_COLS = [
    "tem2", "rhu2", "hcc", "mcc", "lcc", "prs2", "tcc", "snow", "ws2", "sr", 
    "install_power", "hour_sin", "hour_cos", "month_sin", "month_cos",
    "theory_power", "history_power",
    "wd2_sin", "wd2_cos", "sr_rolling_std", "tem2_diff", "theory_rolling_mean",
    "cluster_mean_sr", "cluster_mean_tem2"
]

# FEATURE_COLS = [
#     "tem2", "wd2", "rhu2", "hcc", "mcc", "lcc", "prs2", "tcc", "snow", "ws2", "sr"
# ]
TARGET_COL = "power_residual"
RAW_TARGET_COL = "actual_power"


def _read_csv_with_fallback(data_path: str) -> pd.DataFrame:
    encodings = ["utf-8", "gbk", "gb2312", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(data_path, encoding=enc, low_memory=False)
        except UnicodeDecodeError as exc:
            last_err = exc
    if last_err is not None:
        raise last_err
    return pd.read_csv(data_path, low_memory=False)


def load_and_clean_dataframe(
    data_path: str,
    site_col: str = "location_new",
    time_col: str = "data_time",
    target_col: str = RAW_TARGET_COL,
    night_sr_threshold: float = 1e-6,
    expected_freq: str = "15min",
) -> pd.DataFrame:
    df = _read_csv_with_fallback(data_path)

    if site_col not in df.columns or time_col not in df.columns:
        raise ValueError(f"CSV缺少必要列: {site_col} 或 {time_col}")

    # 时间与站点列标准化
    # df[site_col] = pd.to_numeric(df[site_col], errors="coerce")
    # df[time_col] = pd.to_datetime(df[time_col], format="%Y/%m/%d %H:%M", errors="coerce")
    # df = df.dropna(subset=[site_col, time_col]).copy()
    # df[site_col] = df[site_col].astype(int)

    # 时间与站点列标准化
    df[site_col] = pd.to_numeric(df[site_col], errors="coerce")
    df[time_col] = pd.to_datetime(df[time_col], format="%Y/%m/%d %H:%M", errors="coerce")
    df = df.dropna(subset=[site_col, time_col]).copy()
    df[site_col] = df[site_col].astype(int)

    # === 优化 2：提取时间特征并归一化 ===
    df['hour'] = df[time_col].dt.hour + df[time_col].dt.minute / 60.0
    df['month'] = df[time_col].dt.month
    
    # 周期性编码 (正弦/余弦变换是处理时间序列的最佳实践)
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)

    # === 新增：风向的连续周期性编码 ===
    df['wd2'] = pd.to_numeric(df['wd2'], errors='coerce').fillna(0.0)
    df['wd2_sin'] = np.sin(2 * np.pi * df['wd2'] / 360.0)
    df['wd2_cos'] = np.cos(2 * np.pi * df['wd2'] / 360.0)

    # 数值列清理 (移动到使用 diff 和 rolling 之前，确保存是浮点数)
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # === 新增：气象突变预警特征 ===
    # 1. 辐照度在过去两小时（8个时间步）内的波动程度（标准差）
    df['sr_rolling_std'] = df['sr'].rolling(window=8, min_periods=1).std().fillna(0)
    
    # 2. 温度的一阶差分（当前时刻减去上一时刻，捕捉突变降温）
    df['tem2_diff'] = df['tem2'].diff(1).fillna(0)
    
    # 3. 理论功率的滚动均值（平滑掉极端物理边界）
    if 'theory_power' in df.columns:
        df['theory_rolling_mean'] = df['theory_power'].rolling(window=4, min_periods=1).mean().fillna(0)

    # 将目标列转成数值，过滤中文说明行等异常行
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df[target_col] = df[target_col].clip(lower=0.0)

    # 【新增代码】：处理理论功率
    df["theory_power"] = pd.to_numeric(df["theory_power"], errors="coerce").fillna(0.0)

    # 【新增代码】：处理历史功率
    # df["history_power"] = pd.to_numeric(df["history_power"], errors="coerce").fillna(0.0)

    # 夜间处理：将异常微小负功率截断为0，保留夜间样本帮助模型学习昼夜节律
    df[target_col] = df[target_col].clip(lower=0.0)

    # === 新增：将绝对功率转化为标幺值，但在转化前保留真实功率，以便后续能够完全反推 ===
    # 提取 install_power 并保存
    df["install_power"] = pd.to_numeric(df["install_power"], errors="coerce")
    
    # 将功率进行标幺化（限定在 0~1 左右）
    df[target_col] = df[target_col] / (df["install_power"] + 1e-6)
    df["theory_power"] = df["theory_power"] / (df["install_power"] + 1e-6)

    # 我们把处理好的实发标幺功率，复制一份作为历史功率给 LSTM 学习
    df["history_power"] = df[target_col].copy()

    # 数值列已经在前面清理过了，这里不再重复
    
    # 分站点补齐时间频率 + 插值，避免部分时间点缺失
    filled_frames = []
    for site_id, g in df.groupby(site_col):
        g = g.sort_values(time_col).drop_duplicates(subset=[time_col], keep="last")

        full_time_index = pd.date_range(g[time_col].min(), g[time_col].max(), freq=expected_freq)
        g = g.set_index(time_col).reindex(full_time_index)
        g.index.name = time_col
        g = g.reset_index()
        g[site_col] = site_id

        # === 新增：基于物理常识的异常值清洗 ===
        # 此时 target_col 和 history_power 已转化为除以装机容量的标幺值（0-1之间）
        
        # 异常类型1：限电/硬件脱网（高辐照度但功率极小）
        curtailment_mask = (g['sr'] > 500) & (g[target_col] < 0.05)
        # 异常类型2：传感器夜间漂移/故障（极低辐照度但功率异常）
        fault_mask = (g['sr'] < 10) & (g[target_col] > 0.2)
        
        # 将异常点设为 NaN 以便后续插值或丢弃
        g.loc[curtailment_mask | fault_mask, [target_col, 'history_power']] = np.nan

        # 获取在当前 DataFrame 中确实存在的基础特征列，排除尚未计算的集群特征
        available_cols = [c for c in FEATURE_COLS if c in g.columns]
        num_cols = available_cols + [target_col]
        
        # === 优化：限制暴力插值的距离，宁缺毋滥（limit=4 即最多向后/向前补1小时）===
        g[num_cols] = g[num_cols].interpolate(method="linear", limit=4, limit_direction="both")
        
        # 移除 ffill 和 bfill，直接 dropna，避免连续缺失处生成“伪造的平滑平移特征”
        # 这里只对 available_cols + target_col 做 dropna
        g = g.dropna(subset=num_cols)

        # 夜间合理处理：无辐照时目标功率置0，气象特征保留插值结果
        g["sr"] = g["sr"].clip(lower=0.0)
        g.loc[g["sr"] <= night_sr_threshold, target_col] = 0.0
        g[target_col] = g[target_col].clip(lower=0.0)

        # 【在这里加一行】：确保历史功率和彻底清洗干净的目标功率完全一致！
        g["history_power"] = g[target_col]
        
        filled_frames.append(g)

    df = pd.concat(filled_frames, ignore_index=True)
    df = df.sort_values([site_col, time_col]).reset_index(drop=True)

    # === 新增：计算集群空间特征 ===
    # 按照时间列进行groupby，计算同一时刻所有场站的平均辐照度(sr)和平均温度(tem2)
    cluster_features = df.groupby(time_col)[['sr', 'tem2']].mean().reset_index()
    cluster_features.rename(columns={'sr': 'cluster_mean_sr', 'tem2': 'cluster_mean_tem2'}, inplace=True)
    
    # 将这两列新生成的集群特征拼接到原本的df上
    df = pd.merge(df, cluster_features, on=time_col, how='left')

    # === 新增：构建残差学习目标 ===
    # 真实功率已经和理论功率都是除以了装机容量的标幺值
    # 我们预测的目标变为残差 (Residual) = 理论功率 - 实际功率
    df[TARGET_COL] = df['theory_power'] - df[target_col]

    return df
