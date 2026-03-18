import pandas as pd
import numpy as np

def generate_cooling_comparison(Q_kW, delta_t_list=[3, 4, 5], C=120):
    """
    Q_kW: 总供冷量 (kW)
    delta_t_list: 待测试的温差列表 (℃)
    C: 海澄-威廉系数 (钢管通常取120)
    """
    # 物理常数
    rho = 1000  # 水密度 kg/m3
    cp = 4.187  # 水比热容 kJ/(kg·℃)
    g = 9.81
    
    # 扩展管径数据 (DN300 - DN1200)
    dn_list = [300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1200]
    
    table_data = []
    
    for dt in delta_t_list:
        # 计算该温差下的质量流量 (kg/s) 和 体积流量 (m3/h)
        # Q = G * cp * dt
        G_kg_s = Q_kW / (cp * dt)
        G_m3h = (G_kg_s / rho) * 3600
        G_m3s = G_kg_s / rho
        
        for dn in dn_list:
            d = dn / 1000.0  # 假设内径近似等于公称直径
            
            # 计算流速 v (m/s)
            v = (4 * G_m3s) / (np.pi * d**2)
            
            # 计算比摩阻 R (Pa/m) - 海澄威廉公式
            R = 105.17 * (G_m3s / C)**1.852 * d**(-4.87) * rho * g
            
            # 筛选工程合理的流速范围 (0.7 - 3.5 m/s) 以免表格过长
            if 0.7 <= v <= 3.8:
                table_data.append({
                    "温差(℃)": dt,
                    "总流量(m³/h)": round(G_m3h, 1),
                    "管径(DN)": dn,
                    "流速(m/s)": round(v, 2),
                    "比摩阻(Pa/m)": round(R, 1),
                    "每公里压降(kPa)": round(R * 1000 / 1000, 1)
                })
    
    # 转换为 DataFrame 并输出
    df = pd.DataFrame(table_data)
    return df

# --- 设置参数并运行 ---
# 假设当前能源站供冷规模为 25,000 kW
results_df = generate_cooling_comparison(Q_kW=2896)

# 打印美化后的表格
print(results_df.to_string(index=False))