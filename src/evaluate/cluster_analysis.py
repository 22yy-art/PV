import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_station_correlation():
    # 1. 确定数据路径 (请根据你的实际运行目录微调)
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "pv_data.csv"
    
    print("正在加载数据...")
    # 使用 gbk 编码，并且用 skiprows=[1] 直接跳过第二行的中文翻译行
    df = pd.read_csv(data_path, encoding='gbk', skiprows=[1], low_memory=False)
    
    print("正在清洗数据格式...")
    # 强制将我们需要的列转换成浮点数，遇到解析不了的脏数据直接变成 NaN
    df['actual_power'] = pd.to_numeric(df['actual_power'], errors='coerce')
    df['location_new'] = pd.to_numeric(df['location_new'], errors='coerce')
    
    # 2. 数据透视 (Pivot)
    # 目标：让每一行是一个时间点，每一列是一个场站 (location_new)，值是实际功率 (actual_power)
    # 这样就能构成多条平行的时序曲线
    print("正在构建时序矩阵...")
    pivot_df = df.pivot_table(index='data_time', columns='location_new', values='actual_power')
    
    # 填补可能存在的极少数缺失值（用前向填充）
    pivot_df = pivot_df.fillna(method='ffill').fillna(method='bfill')
    
    # 3. 计算皮尔逊相关系数矩阵 (Pearson Correlation)
    # 这与论文中的互信息起到了极度相似的拓扑衡量作用，且计算速度极快
    print("正在计算场站间的空间相关系数矩阵...")
    corr_matrix = pivot_df.corr(method='pearson')
    
    # 4. 寻找最高度同步的“神秘场站对”
    # 将对角线（自己和自己相关，系数必然为1）设为 NaN，方便寻找真正的相关性最大值
    np.fill_diagonal(corr_matrix.values, np.nan)
    
    # 将矩阵展平并降序排列，找出最相关的 Top 5
    corr_unstacked = corr_matrix.unstack().dropna().sort_values(ascending=False)
    # 剔除重复的组合 (比如 A-B 和 B-A)
    corr_unstacked = corr_unstacked[corr_unstacked.index.get_level_values(0) < corr_unstacked.index.get_level_values(1)]
    
    print("\n=== 💡 发现高度同步的光伏场站对 TOP 5 ===")
    for (station_a, station_b), corr_value in corr_unstacked.head(5).items():
        print(f"场站 {station_a:2.0f} 与 场站 {station_b:2.0f} 的相关系数高达: {corr_value:.4f}")
        
    # 5. 绘制与论文高度相似的高级热力图 (答辩 PPT 素材)
    print("\n正在绘制空间集群热力图...")
    plt.figure(figsize=(10, 8))
    # 设置中文字体（防止乱码），如果没有黑体可以去掉或换成其他字体
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False
    
    sns.heatmap(corr_matrix, annot=False, cmap='Reds', square=True, 
                cbar_kws={'label': 'Pearson Correlation (皮尔逊相关系数)'})
    
    plt.title('光伏集群空间互相关性热力图 (Spatial Correlation Heatmap)', fontsize=16)
    plt.xlabel('场站编号 (location_new)', fontsize=12)
    plt.ylabel('场站编号 (location_new)', fontsize=12)
    
    # 调整布局并保存
    plt.tight_layout()
    output_path = project_root / "outputs" / "plots" / "station_correlation_heatmap.png"
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"🎉 热力图已成功保存至: {output_path}")

if __name__ == '__main__':
    analyze_station_correlation()