import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def parse_energy_report(file_path, sheet_name=0):
    """解析报表，返回瞬时流量>0的数据及列名"""
    raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None, dtype=str)
    
    # 定位表头行
    header_row_idx = None
    for i, row in raw.iterrows():
        row_vals = row.astype(str).str.strip()
        if '区域名称' in row_vals.values:
            header_row_idx = i
            break
    if header_row_idx is None:
        raise ValueError("未找到包含'区域名称'的表头行")
    
    main_header = raw.iloc[header_row_idx].fillna('').astype(str).str.strip()
    sub_header = None
    if header_row_idx + 1 < len(raw):
        sub_candidate = raw.iloc[header_row_idx + 1].fillna('').astype(str).str.strip()
        if any('瞬时值' in v or '累计值' in v or '瞬时流量' in v for v in sub_candidate):
            sub_header = sub_candidate
    
    # 合并列名
    new_columns = []
    for i, main in enumerate(main_header):
        if sub_header is not None and sub_header[i] != '':
            col_name = f"{main}_{sub_header[i]}" if main != '' else sub_header[i]
        else:
            col_name = main
        col_name = col_name.replace('（', '(').replace('）', ')').strip()
        new_columns.append(col_name)
    
    data_start = header_row_idx + 1
    if sub_header is not None:
        data_start += 1
    df_data = raw.iloc[data_start:].copy()
    df_data.columns = new_columns
    df_data.dropna(how='all', inplace=True)
    df_data.reset_index(drop=True, inplace=True)
    
    # 定位关键列
    flow_col = None
    supply_col = None
    return_col = None
    for col in df_data.columns:
        if '瞬时流量' in col or ('流量' in col and '瞬时' in col):
            flow_col = col
        if '供水温度' in col:
            supply_col = col
        if '回水温度' in col:
            return_col = col
    
    if not flow_col or not supply_col or not return_col:
        raise ValueError(f"未找到所需列，实际列名：{list(df_data.columns)}")
    
    # 转换数值
    df_data[flow_col] = pd.to_numeric(df_data[flow_col], errors='coerce')
    df_data[supply_col] = pd.to_numeric(df_data[supply_col], errors='coerce')
    df_data[return_col] = pd.to_numeric(df_data[return_col], errors='coerce')
    
    # 过滤瞬时流量 > 0 且温度非空
    mask = (df_data[flow_col] > 0) & df_data[supply_col].notna() & df_data[return_col].notna()
    df_filtered = df_data[mask].copy()
    
    if len(df_filtered) == 0:
        print("警告：无有效数据")
        return df_filtered, {}
    
    return df_filtered, {'flow': flow_col, 'supply': supply_col, 'return': return_col}


def plot_temperature_distribution(df, supply_col, return_col, save_path='temp_distribution.png'):
    """绘制概率分布图（样本分布，未加权）"""
    supply = df[supply_col]
    return_temp = df[return_col]
    
    avg_supply_simple = supply.mean()
    avg_return_simple = return_temp.mean()
    
    plt.figure(figsize=(12, 5))
    
    # 左图：概率密度直方图 + KDE
    plt.subplot(1, 2, 1)
    sns.histplot(supply, bins=20, kde=True, stat='probability', label=f'供水温度 (算术均值={avg_supply_simple:.1f}℃)', alpha=0.6)
    sns.histplot(return_temp, bins=20, kde=True, stat='probability', label=f'回水温度 (算术均值={avg_return_simple:.1f}℃)', alpha=0.6)
    plt.xlabel('温度 (℃)')
    plt.ylabel('概率密度')
    plt.title('供水与回水温度概率分布 (样本分布)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 右图：累积分布函数
    plt.subplot(1, 2, 2)
    supply_sorted = np.sort(supply)
    return_sorted = np.sort(return_temp)
    supply_cdf = np.arange(1, len(supply_sorted)+1) / len(supply_sorted)
    return_cdf = np.arange(1, len(return_sorted)+1) / len(return_sorted)
    plt.plot(supply_sorted, supply_cdf, label='供水温度 CDF', linewidth=2)
    plt.plot(return_sorted, return_cdf, label='回水温度 CDF', linewidth=2)
    plt.xlabel('温度 (℃)')
    plt.ylabel('累积概率')
    plt.title('累积分布函数 (CDF)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"概率分布图已保存为: {save_path}")


def main():
    file_path = r'C:\Users\ASUS\Downloads\冷热量表历史记录报表20260427173139.xls'  # 请修改为实际路径
    
    try:
        df, cols = parse_energy_report(file_path)
        if df.empty:
            return
        
        flow_col = cols['flow']
        supply_col = cols['supply']
        return_col = cols['return']
        
        # 加权平均温度（以瞬时流量为权重）
        total_flow = df[flow_col].sum()
        weighted_avg_supply = (df[supply_col] * df[flow_col]).sum() / total_flow
        weighted_avg_return = (df[return_col] * df[flow_col]).sum() / total_flow
        
        # 算术平均（对比用）
        simple_avg_supply = df[supply_col].mean()
        simple_avg_return = df[return_col].mean()
        
        print(f"有效样本数: {len(df)}")
        print(f"总瞬时流量（权重和）: {total_flow:.2f} m³/h")
        print("\n=== 加权平均温度（以瞬时流量为权重）===")
        print(f"加权平均供水温度: {weighted_avg_supply:.2f} ℃")
        print(f"加权平均回水温度: {weighted_avg_return:.2f} ℃")
        print("\n=== 算术平均温度（对比）===")
        print(f"算术平均供水温度: {simple_avg_supply:.2f} ℃")
        print(f"算术平均回水温度: {simple_avg_return:.2f} ℃")
        
        # 绘制概率分布图（样本分布）
        plot_temperature_distribution(df, supply_col, return_col, save_path='供水回水温度分布.png')
        
    except Exception as e:
        print(f"处理出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()