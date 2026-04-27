
import pulp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 解决 Matplotlib 中文显示乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC']  # 优先使用系统自带中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示坐标轴上的负号

# ==========================================
# 1. 真实数据读取与全局参数设定 (已修改为单 Excel 多 Sheet 读取)
# ==========================================
print("正在读取 Excel 设备与参数配置...")
excel_path = 'D:/minicondadaima/lianxi/中心站测试模板.xlsx'  # 确保文件名与您本地一致

# 读取同一个 Excel 文件的不同 Sheet
load_df = pd.read_excel(excel_path, sheet_name='负荷')
chiller_df = pd.read_excel(excel_path, sheet_name='电制冷机')
ice_chiller_df = pd.read_excel(excel_path, sheet_name='蓄冰机')
price_df = pd.read_excel(excel_path, sheet_name='电价')
params_df = pd.read_excel(excel_path, sheet_name='参数').set_index('key')

# 基础天负荷与电价曲线
BASE_LOAD = load_df['load_kwc'].values
TOU_PRICES = price_df['price'].values

# 系统核心参数
DAYS_IN_MONTH = int(params_df.loc['days_in_month', 'value'])
HOURS_IN_DAY = 24
TOTAL_HOURS = DAYS_IN_MONTH * HOURS_IN_DAY
DEMAND_CHARGE_RATE = params_df.loc['demand_price', 'value'] # 48元/kVA
PF = params_df.loc['pf', 'value']                           # 功率因数 0.85
M_PENALTY = 10000.0                                         # 软约束天价罚款

# 物理设备能力约束提取
CHILLER1_QMAX = chiller_df.loc[0, 'Qmax']
CHILLER1_COP = chiller_df.loc[0, 'COP_100']

CHILLER2_QMAX = chiller_df.loc[1, 'Qmax']
CHILLER2_COP = chiller_df.loc[1, 'COP_100']

ICE_MAKE_QMAX = ice_chiller_df.loc[0, 'Q_charge_max']
ICE_MAKE_COP = ice_chiller_df.loc[0, 'COP']
TANK_CAPACITY_MAX = ice_chiller_df.loc[0, 'E_ice_max']
MAX_DISCHARGE_PER_HOUR = TANK_CAPACITY_MAX * ice_chiller_df.loc[0, 'discharge_ratio']


# ==========================================
# 2. 生成 30 天逐时负荷并按比例切分 (供 Simulink 使用)
# ==========================================
print("正在生成 30 天 (720小时) 管网水力负荷边界条件...")
np.random.seed(42)
MONTHLY_LOAD = []
for day in range(DAYS_IN_MONTH):
    # 模拟天气波动：每天在基准负荷上产生 ±10% 的随机浮动
    daily_variation = np.random.uniform(0.9, 1.1, size=HOURS_IN_DAY)
    MONTHLY_LOAD.append(BASE_LOAD * daily_variation)

# 展平为连续的 720 小时总负荷
flat_total_load = np.concatenate(MONTHLY_LOAD)

# 按照 Simulink 模型的 1:9:12 比例分配末端负荷 (总份数 22)
load_u3 = flat_total_load * (1 / 22)
load_u4 = flat_total_load * (9 / 22)
load_u6 = flat_total_load * (12 / 22)

# 导出为 Simulink From Workspace 可直接读取的格式 [time_sec, value]
time_series_sec = np.arange(TOTAL_HOURS) * 3600  
simulink_load_df = pd.DataFrame({
    'time_sec': time_series_sec,
    'total_load': flat_total_load,
    'user3_load': load_u3,
    'user4_load': load_u4,
    'user6_load': load_u6
})
simulink_load_df.to_csv('Simulink_30Days_Load.csv', index=False)
print(" 物理边界已生成！文件已保存为 'Simulink_30Days_Load.csv'")

# ==========================================
# 3. 内环：单日日程调度优化器 (PuLP)
# ==========================================
def optimize_daily_dispatch(day_idx, initial_ice, p_target_kva, daily_load):
    prob = pulp.LpProblem(f"Daily_Optimization_Day_{day_idx}", pulp.LpMinimize)
    
    # --- 决策变量 ---
    Q_chiller1 = pulp.LpVariable.dicts("Q_chiller1", range(HOURS_IN_DAY), lowBound=0, upBound=CHILLER1_QMAX)
    Q_chiller2 = pulp.LpVariable.dicts("Q_chiller2", range(HOURS_IN_DAY), lowBound=0, upBound=CHILLER2_QMAX)
    Q_ice_discharge = pulp.LpVariable.dicts("Q_ice_discharge", range(HOURS_IN_DAY), lowBound=0, upBound=MAX_DISCHARGE_PER_HOUR)
    Q_ice_charge = pulp.LpVariable.dicts("Q_ice_charge", range(HOURS_IN_DAY), lowBound=0, upBound=ICE_MAKE_QMAX)
    
    Tank_SOC = pulp.LpVariable.dicts("Tank_SOC", range(HOURS_IN_DAY + 1), lowBound=0, upBound=TANK_CAPACITY_MAX)
    P_excess_kva = pulp.LpVariable.dicts("P_excess_kva", range(HOURS_IN_DAY), lowBound=0, cat='Continuous')
    
    # --- 约束条件 ---
    prob += Tank_SOC[0] == initial_ice
    
    for t in range(HOURS_IN_DAY):
        # 1. 节点能量守恒 (冷机1 + 冷机2 + 融冰 == 终端冷负荷)
        prob += Q_chiller1[t] + Q_chiller2[t] + Q_ice_discharge[t] == daily_load[t]
        
        # 2. 冰罐时序状态转移
        prob += Tank_SOC[t+1] == Tank_SOC[t] + Q_ice_charge[t] - Q_ice_discharge[t]
        
        # 3. 电功率与视在功率转换 (KW 转 KVA)
        P_kw = (Q_chiller1[t]/CHILLER1_COP) + (Q_chiller2[t]/CHILLER2_COP) + (Q_ice_charge[t]/ICE_MAKE_COP) 
        P_pump_kw = 50 + 0.02 * daily_load[t] # 水泵预估功率占位符，后续可替换为水力管网方程
        Total_Power_kva = (P_kw + P_pump_kw) / PF 
        
        # 4. 容量费软约束拦截
        prob += Total_Power_kva <= p_target_kva + P_excess_kva[t]

    # --- 目标函数 ---
    daily_energy_cost = pulp.lpSum([
        (((Q_chiller1[t]/CHILLER1_COP) + (Q_chiller2[t]/CHILLER2_COP) + (Q_ice_charge[t]/ICE_MAKE_COP) + 50 + 0.02*daily_load[t]) * TOU_PRICES[t]) 
        for t in range(HOURS_IN_DAY)
    ])
    penalty_cost = pulp.lpSum([P_excess_kva[t] * M_PENALTY for t in range(HOURS_IN_DAY)])
    
    prob += daily_energy_cost + penalty_cost
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    cost_val = pulp.value(daily_energy_cost)
    penalty_val = pulp.value(penalty_cost)
    excess_val = sum([P_excess_kva[t].varValue for t in range(HOURS_IN_DAY)])
    
    return cost_val, penalty_val, excess_val, Tank_SOC[HOURS_IN_DAY].varValue

# ==========================================
# 4. 外环：需量扫描与全月博弈主程序
# ==========================================
def scan_monthly_demand():
    # 根据装机功率和0.85功率因数，设定合适的 KVA 扫描区间
    p_targets_to_test = range(800, 2500, 100) 
    results = []
    
    print("\n🚀 启动外环容量阈值扫描 (基于 30 天动态负荷)...")
    print("-" * 75)
    print(f"{'P_target(kVA)':<14} | {'月电度电费(元)':<13} | {'容量费(元)':<12} | {'总成本(元)':<12} | {'罚款(元)'}")
    print("-" * 75)
    
    for p_target in p_targets_to_test:
        monthly_energy_cost, monthly_penalty_cost, total_excess = 0, 0, 0
        current_ice_soc = 0.0 
        
        for day in range(DAYS_IN_MONTH):
            cost, penalty, excess, next_ice = optimize_daily_dispatch(
                day, current_ice_soc, p_target, MONTHLY_LOAD[day]
            )
            monthly_energy_cost += cost
            monthly_penalty_cost += penalty
            total_excess += excess
            current_ice_soc = next_ice 
            
        demand_charge = p_target * DEMAND_CHARGE_RATE
        total_cost = monthly_energy_cost + demand_charge
        
        results.append({
            'P_target': p_target, 'Energy_Cost': monthly_energy_cost,
            'Demand_Charge': demand_charge, 'Total_Cost': total_cost,
            'Penalty': monthly_penalty_cost, 'Is_Feasible': total_excess < 1.0 
        })
        print(f"{p_target:<14} | {monthly_energy_cost:<13.1f} | {demand_charge:<12.1f} | {total_cost:<12.1f} | {monthly_penalty_cost:.1f}")

    # ==========================================
    # 5. 寻优结果输出与绘图
    # ==========================================
    df = pd.DataFrame(results)
    feasible_df = df[df['Is_Feasible'] == True].copy()
    
    if feasible_df.empty:
        print("\n❌ 警告：所有容量阈值均无法满足要求！请检查冷机容量或放宽 KVA 扫描上限。")
        return
        
    optimal_row = feasible_df.loc[feasible_df['Total_Cost'].idxmin()]
    optimal_P = optimal_row['P_target']
    
    print("-" * 75)
    print(f"✅ 寻优完成！最优月度报装需量为: ** {optimal_P} kVA **")
    print(f"预计月度总账单为: {optimal_row['Total_Cost']:.1f} 元")
    print("-" * 75)
    
    plt.figure(figsize=(10, 6))
    plt.plot(feasible_df['P_target'], feasible_df['Total_Cost'], marker='o', label='Total Cost (总账单)', color='purple')
    plt.plot(feasible_df['P_target'], feasible_df['Energy_Cost'], linestyle='--', label='Energy Cost (谷峰电度费)', color='blue')
    plt.plot(feasible_df['P_target'], feasible_df['Demand_Charge'], linestyle='--', label='Demand Charge (容量费)', color='orange')
    plt.axvline(x=optimal_P, color='red', linestyle='-.', label=f'Optimal: {optimal_P} kVA')
    
    plt.title('Capacity Demand (kVA) vs. Monthly Cost Optimization')
    plt.xlabel('Target Maximum Apparent Power (kVA)')
    plt.ylabel('Monthly Cost (RMB)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('optimization_result.png')

# ==========================================
# ==========================================
# 全新阶段：基于 Simulink 真实冷量反馈的联合调度架构
# ==========================================
import os
import scipy.io as sio # 用于读取 Matlab 数据


def step1_generate_blind_pressure_for_simulink():
    print("\n🌊 正在基于真实水泵铭牌 (扬程46m) 生成水压边界...")
    pump_pressure_seq = []
    
    # --- 真实水泵物理参数映射 ---
    P_MAX_PA = 451000.0  # 46m 扬程对应的满载管网压差 (Pa)
    P_MIN_PA = 200000.0  # 变频器 30Hz 限制下的保底压差 (Pa)
    
    for day in range(DAYS_IN_MONTH):
        daily_load = MONTHLY_LOAD[day]
        for t in range(HOURS_IN_DAY):
            # 负荷率
            load_ratio = daily_load[t] / max(BASE_LOAD)
            
            # 遵循【水泵相似定律】的二次方抛物线变频方程
            # 即: 水泵压差与流量(负荷)的平方成正比，但不得低于变频器保底频率对应的压力
            pressure_pa = max(P_MIN_PA, P_MAX_PA * (load_ratio ** 2)) 
            
            pump_pressure_seq.append(pressure_pa)
            
    time_series_sec = np.arange(TOTAL_HOURS) * 3600
    df_pressure = pd.DataFrame({
        'time_sec': time_series_sec,
        'pump_pressure': pump_pressure_seq
    })
    df_pressure.to_csv('Simulink_30Days_Commands.csv', index=False)
    print("✅ 真实变频水泵水力指令已生成！")
def step2_optimize_with_real_physics_data():
    """
    第二步：拿到 Simulink 反馈的真实冷量需求，做最终经济排产
    """
    # 1. 读取 Simulink 跑出来的真实冷量数据
    if not os.path.exists('sim_result.mat'):
        print("❌ 找不到 sim_result.mat！请先去 MATLAB 里把跑完的数据保存下来！")
        return
        
    print("\n📥 正在读取物理管网反馈的【真实冷量需求曲线】...")
    mat_data = sio.loadmat('sim_result.mat')
    

    try:
        # 因为在 MATLAB 里单独保存了，所以直接通过键值读取，不需要加 ['out']
        # 注意要用 flatten() 把矩阵展平为一维数组
        real_q_demand_array = mat_data['Q_real_demand'].flatten()
        real_time_sec = mat_data['tout'].flatten()
    except KeyError:
        print("❌ 数据结构不对，请检查 MATLAB 的 save 指令是否正确执行！")
        return
    # 2. 将连续的秒级数据，降采样（平均）为 720 小时的逐时数据供 PuLP 使用
    # (这里用一个极简的切片平均逻辑)
    hourly_real_demand = []
    points_per_hour = len(real_time_sec) // TOTAL_HOURS
    for i in range(TOTAL_HOURS):
        start_idx = i * points_per_hour
        end_idx = (i+1) * points_per_hour
        avg_q = np.mean(real_q_demand_array[start_idx:end_idx])
        hourly_real_demand.append(avg_q)
        
    print("✅ 物理冷量提取成功！准备开展基于真实数据的经济寻优...")
    
    # 3. 将真实需求注入 PuLP 进行排产 (以 1400 kVA 为例)
    optimal_p_target = 1400
    current_ice_soc = 0.0
    total_cost = 0
    
    for day in range(DAYS_IN_MONTH):
        # ⚠️ 注意：这里传入的不再是理想的 daily_load，而是物理引擎算出的真实需求！
        real_daily_load = hourly_real_demand[day*24 : (day+1)*24] 
        cost, pen, exc, next_ice = optimize_daily_dispatch(
            day, current_ice_soc, optimal_p_target, real_daily_load
        )
        total_cost += cost
        current_ice_soc = next_ice
        
    print("-" * 50)
    print(f"🎉 终极物理-经济联合测算完成！")
    print(f"在 {optimal_p_target} kVA 限制下，考虑管网物理损耗后的真实月电费为: {total_cost:.1f} 元")
    print("-" * 50)


# 执行入口
if __name__ == "__main__":
    # 请先取消注释运行 step1，去 Simulink 跑完保存结果后，再注释掉 step1，运行 step2
    #step1_generate_blind_pressure_for_simulink()
    step2_optimize_with_real_physics_data()