import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

def plot_feature_importance(data_path):
    # 1. 加载数据
    df = None
    for enc in ["utf-8", "gbk", "gb2312", "latin1"]:
        try:
            df = pd.read_csv(data_path, encoding=enc, low_memory=False)
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        raise ValueError(f"无法使用常见编码读取文件: {data_path}")
    
    # 定义特征和目标变量
    features = ["tem2", "wd2", "rhu2", "hcc", "mcc", "lcc", "prs2", "tcc", "snow", "ws2", "sr"]
    target = "actual_power"
    
    # 数据清洗：转数值并去掉缺失值
    for col in features + [target]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=features + [target])

    X = df[features]
    y = df[target]

    # 为了加快速度，如果数据太大可以随机采样 50000 条
    if len(X) > 50000:
        idx = np.random.choice(len(X), 50000, replace=False)
        X = X.iloc[idx]
        y = y.iloc[idx]

    # 2. 训练随机森林模型获取权重
    print("正在计算特征权重，请稍候...")
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # 3. 提取权重并排序
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [features[i] for i in indices]
    sorted_weights = importances[indices]

    # 4. 图形化展示
    plt.figure(figsize=(10, 6))
    
    # 【修复点 1】：添加了 hue=sorted_features 和 legend=False，解决 FutureWarning 警告
    sns.barplot(x=sorted_weights, y=sorted_features, hue=sorted_features, palette="viridis", legend=False)
    
    # 增加数值标签
    for i, v in enumerate(sorted_weights):
        plt.text(v + 0.01, i, f"{v:.3f}", va='center')
        
    plt.title("Meteorological Feature Importance (Random Forest)", fontsize=14)
    plt.xlabel("Importance Weight", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.tight_layout()
    
    # 【修复点 2】：路径字符串前面加了 r，解决 OSError 乱码报错
    plt.savefig(r"E:\PV\outputs\plots\feature_importance.png", dpi=300)
    print("特征权重图已成功保存！")
    plt.show()

if __name__ == "__main__":
    # 【修复点 3】：读取数据的路径字符串前面也加了 r，防止潜在的转义报错
    plot_feature_importance(r"E:\PV\data\pv_data.csv")
