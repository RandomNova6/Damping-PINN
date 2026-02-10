import pandas as pd
import numpy as np
import random

class DataWriter():
    def __init__(self, filepath):
        self.filepath = filepath  

    def save_data(self, df):
        # 确保列名正确
        df.columns = ['theta', 'T', 'phi']
        df.to_excel(self.filepath, index=False)
        return df

def thin_data_logic(df, n, mode='skip_n_delete_1'):
    """
    数据稀疏化函数
    """
    df_reset = df.reset_index(drop=True)
    total_len = len(df_reset)
    
    if mode == 'skip_n_delete_1':
        cycle = n + 1
        drop_indices = [i for i in range(total_len) if (i + 1) % cycle == 0]
        new_df = df_reset.drop(drop_indices).reset_index(drop=True)
        print(f"模式: 隔 {n} 个删 1 个 | 原始: {total_len} -> 删除: {len(drop_indices)} -> 剩余: {len(new_df)}")
        return new_df
        
    elif mode == 'uniform':
        new_df = df_reset.iloc[::n, :].reset_index(drop=True)
        print(f"模式: 均匀采样(步长{n}) | 原始: {total_len} -> 剩余: {len(new_df)}")
        return new_df

def inject_anomalies(df, num_points=3, max_dev=0.2, target_cols=None):
    """
    异常数据注入函数
    :param df: 输入的 DataFrame
    :param num_points: 要修改的数据点数量 (随机选择行)
    :param max_dev: 最大偏差百分比 (例如 0.2 代表 +/- 20%)
    :param target_cols: 指定要注入异常的列名列表 (默认为所有数值列)
    :return: 注入异常后的新 DataFrame
    """
    df_noisy = df.copy()
    rows = len(df_noisy)
    
    if rows == 0:
        return df_noisy

    # 如果没有指定列，默认对 theta, T, phi 进行操作
    if target_cols is None:
        target_cols = ['theta', 'T', 'phi']
    
    # 确保 num_points 不超过总行数
    actual_num = min(num_points, rows)
    
    # 随机选择要修改的行索引
    random_indices = random.sample(range(rows), actual_num)
    
    print("-" * 30)
    print(f"开始注入异常数据 (共 {actual_num} 处, 偏差范围 +/- {max_dev:.0%}):")
    
    for idx in random_indices:
        # 随机选择某一列
        col = random.choice(target_cols)
        
        # 生成随机偏差因子：范围在 [1 - max_dev, 1 + max_dev] 之间
        # random.uniform(-0.2, 0.2) 生成 -0.2 到 0.2 之间的浮点数
        deviation_pct = random.uniform(-max_dev, max_dev)
        factor = 1 + deviation_pct
        
        original_val = df_noisy.at[idx, col]
        new_val = original_val * factor
        
        # 更新数据
        df_noisy.at[idx, col] = new_val
        
        print(f"  -> 行[{idx}], 列[{col}]: {original_val:.4f} -> {new_val:.4f} (变化 {deviation_pct:+.2%})")
        
    return df_noisy

if __name__ == "__main__":
    # 1. 模拟/读取数据 (为了演示方便，如果你的文件不存在，这里创建一个模拟数据)
    try:
        expData17 = pd.read_excel('Experiment17.xlsx', header=None)
        data17_1 = expData17.iloc[:, 0:3].copy()
    except FileNotFoundError:
        print("未找到文件")
    
    data17_1.columns = ['theta', 'T', 'phi']
    print(f"原始数据点数量: {len(data17_1)}")

    # ==========================================
    # 2. 数据稀疏化 (Thinning)
    # ==========================================
    # 场景 B: 均匀采样
    processed_data = thin_data_logic(data17_1, n=1, mode='uniform')

    # ==========================================
    # 3. 注入异常数据 (Anomaly Injection)
    # ==========================================
    # 设定：随机选 3 个点，数值在 +/- 20% 范围内浮动
    # target_cols 可以指定只修改 'theta' 或 'phi'，也可以不传默认全选
    anomalous_data = inject_anomalies(
        processed_data, 
        num_points=0, 
        max_dev=0.20, 
        target_cols=['theta', 'phi'] # 假设我们不想修改时间 T，只修改角度参数
    )

    # ==========================================
    # 4. 保存结果
    # ==========================================
    writer = DataWriter('ExperimentData.xlsx')
    saved_data = writer.save_data(anomalous_data)

    print("-" * 30)
    print("最终保存的数据点 (前5行):")
    print(saved_data.head())
    print(f"最终数量: {len(saved_data)}")