import pandas as pd
import numpy as np

def calculate_pipe_specs(flow_m3s, pipe_material_c=120):
    """
  
    pipe_material_c: 海澄-威廉系数 (碳钢管通常取120，旧管取100)
    """
    # 定义标准 DN (mm) 及其对应的估算内径 (m)
    # 大管径壁厚按标准壁厚估算
    standard_pipes = {
        "DN": [200,250,300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1200],
        "ID_m": [0.200,0.250,0.300, 0.350, 0.400, 0.450, 0.500, 0.600, 0.700, 0.800, 0.900, 1.000, 1.200]
    }
    
    df = pd.DataFrame(standard_pipes)
    G_m3s = flow_m3s
    
    # 1. 计算流速 v = 4G / (pi * D^2)
    df['Velocity (m/s)'] = (4 * G_m3s) / (np.pi * df['ID_m']**2)
    
    # 2. 计算比摩阻 R (Pa/m) - 使用海澄威廉公式
    # hf = 10.67 * L * (G/C)^1.852 * D^-4.87
    # 转化为 Pa/m: R = rho * g * hf
    rho, g = 1000, 9.81
    df['R (Pa/m)'] = 10.67 * (G_m3s / pipe_material_c)**1.852 * df['ID_m']**(-4.87) * rho * g
    
    # 3. 计算每公里的压降 (kPa/km)
    df['DeltaP_per_km (kPa)'] = df['R (Pa/m)'] * 1000 / 1000
    
    # 筛选流速在 0.5 - 4.5 m/s 之间的合理管径
    valid_range = df[(df['Velocity (m/s)'] >= 0.5) & (df['Velocity (m/s)'] <= 4.5)].copy()
    
    return valid_range.round(2)

# --- 示例：假设你的总供冷量对应的流量是 3000 m3/h ---
flow_input = 0.138
result = calculate_pipe_specs(flow_input)

print(f"流量为 {flow_input} m3/h 时的管径选型建议：")
print(result.to_string(index=False))