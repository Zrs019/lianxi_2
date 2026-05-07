
import pulp
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
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

# 固定管径 + 季度固定阀门场景下的管网诊断参数
CP_WATER_KJ_PER_KG_K = 4.186
DESIGN_DELTA_T_C = 5.0
VALVE_SETTING_FILE = 'Valve_Quarter_Settings.csv'
NEXT_VALVE_SETTING_FILE = 'Valve_Quarter_Settings_next.csv'
SIMULINK_BOUNDARY_FILE = 'Simulink_30Days_UserBoundary.csv'
SIMULINK_INPUT_MAT_FILE = 'Simulink_30Days_Input.mat'
VALVE_REPORT_FILE = 'valve_adjustment_report.csv'
STATION_DIAGNOSIS_FILE = 'station_side_diagnosis.txt'
USER_LOAD_RATIOS = {3: 1 / 22, 4: 9 / 22, 6: 12 / 22}
DEFAULT_VALVE_OPENING = {3: 0.40, 4: 0.75, 6: 0.90}
MIN_VALVE_OPENING = 0.10
MAX_VALVE_OPENING = 1.00
VALVE_STEP_LIMIT = 0.15
UNMET_RATIO_TOL = 0.02
LOW_DELTA_T_RATIO = 0.60

# 冷站理想出水温度边界/反馈冷源参数
STATION_SUPPLY_TEMP_SET_C = 7.0
STATION_SUPPLY_TEMP_SET_K = STATION_SUPPLY_TEMP_SET_C + 273.15
STATION_SUPPLY_TEMP_WARN_C = 12.0
STATION_RETURN_TEMP_MAX_C = 20.0
STATION_COOLING_KP_W_PER_K = 1.0e6
STATION_COOLING_CAPACITY_SAFETY_FACTOR = 1.25
STATION_COOLING_CAPACITY_MIN_W = 8.0e6
PUMP_PRESSURE_SOFT_START_SEC = 60.0


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
load_u3 = flat_total_load * USER_LOAD_RATIOS[3]
load_u4 = flat_total_load * USER_LOAD_RATIOS[4]
load_u6 = flat_total_load * USER_LOAD_RATIOS[6]

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
# 5. 固定管径 + 季度固定阀门的 Simulink 边界与结果诊断
# ==========================================
import os
import scipy.io as sio # 用于读取 Matlab 数据


def _user_load_map():
    return {
        3: load_u3,
        4: load_u4,
        6: load_u6,
    }


def load_or_create_valve_settings():
    """
    读取季度阀门开度。如果文件不存在，则按默认开度创建模板。
    valve_opening 使用 0~1 表示，Simulink 中可映射为阀门 Kv 或局部阻力系数。
    """
    if os.path.exists(VALVE_SETTING_FILE):
        settings = pd.read_csv(VALVE_SETTING_FILE)
    else:
        rows = []
        for user_id, ratio in USER_LOAD_RATIOS.items():
            design_load_kw = float(np.max(flat_total_load) * ratio)
            design_flow_kg_s = design_load_kw / (CP_WATER_KJ_PER_KG_K * DESIGN_DELTA_T_C)
            rows.append({
                'user_id': user_id,
                'valve_opening': DEFAULT_VALVE_OPENING[user_id],
                'min_opening': MIN_VALVE_OPENING,
                'max_opening': MAX_VALVE_OPENING,
                'design_delta_t_c': DESIGN_DELTA_T_C,
                'design_flow_kg_s': design_flow_kg_s,
            })
        settings = pd.DataFrame(rows)
        settings.to_csv(VALVE_SETTING_FILE, index=False, encoding='utf-8-sig')
        print(f"✅ 已创建季度阀门开度模板: {VALVE_SETTING_FILE}，请按现场实际开度修正后再跑 Simulink。")

    settings['user_id'] = settings['user_id'].astype(int)
    return settings.set_index('user_id')


def _workspace_series(values):
    """From Workspace 使用的 [time_sec, value] 矩阵。"""
    return np.column_stack((time_series_sec, np.asarray(values, dtype=float)))


def _workspace_series_with_time(times, values):
    """From Workspace 使用的自定义 [time_sec, value] 矩阵。"""
    return np.column_stack((np.asarray(times, dtype=float), np.asarray(values, dtype=float)))


def _constant_workspace_series(value):
    """生成长度与仿真时长一致的常数 From Workspace 矩阵。"""
    return _workspace_series(np.full(TOTAL_HOURS, float(value)))


def _soft_start_series(hourly_values, start_value=0.0, ramp_sec=PUMP_PRESSURE_SOFT_START_SEC):
    """
    给压差源等容易触发初始化断言的输入增加启动斜坡。
    t=0 使用 start_value，ramp_sec 后达到第一个小时值，之后按逐时曲线运行。
    """
    hourly_values = np.asarray(hourly_values, dtype=float)
    if ramp_sec <= 0:
        return _workspace_series(hourly_values)

    times = np.concatenate(([0.0, float(ramp_sec)], time_series_sec[1:]))
    values = np.concatenate(([float(start_value), hourly_values[0]], hourly_values[1:]))
    return _workspace_series_with_time(times, values)


def step1_generate_fixed_valve_boundaries_for_simulink():
    """
    生成供 Simulink 使用的长期边界条件。
    用户负荷是外生需求；阀门开度是季度固定参数，不再随负荷实时改变。
    """
    valve_settings = load_or_create_valve_settings()
    user_loads = _user_load_map()
    total_load_w = flat_total_load * 1000.0
    station_cooling_feedforward_w = total_load_w.copy()
    station_cooling_capacity_w = max(
        float(np.max(station_cooling_feedforward_w) * STATION_COOLING_CAPACITY_SAFETY_FACTOR),
        STATION_COOLING_CAPACITY_MIN_W,
    )
    station_supply_temp_set_k = np.full(TOTAL_HOURS, STATION_SUPPLY_TEMP_SET_K)
    station_supply_temp_set_c = np.full(TOTAL_HOURS, STATION_SUPPLY_TEMP_SET_C)
    station_supply_temp_warn_c = np.full(TOTAL_HOURS, STATION_SUPPLY_TEMP_WARN_C)
    station_return_temp_max_c = np.full(TOTAL_HOURS, STATION_RETURN_TEMP_MAX_C)

    df = pd.DataFrame({
        'time_sec': time_series_sec,
        'total_load': total_load_w,
        'station_supply_temp_set_K': station_supply_temp_set_k,
        'station_supply_temp_set_C': station_supply_temp_set_c,
        'station_supply_temp_warn_C': station_supply_temp_warn_c,
        'station_return_temp_max_C': station_return_temp_max_c,
        'station_cooling_feedforward_W': station_cooling_feedforward_w,
        'station_cooling_capacity_W': np.full(TOTAL_HOURS, station_cooling_capacity_w),
        'station_cooling_kp_W_per_K': np.full(TOTAL_HOURS, STATION_COOLING_KP_W_PER_K),
    })
    mat_inputs = {
        'total_load': _workspace_series(total_load_w),
        'station_supply_temp_set_K': _workspace_series(station_supply_temp_set_k),
        'station_supply_temp_set_C': _workspace_series(station_supply_temp_set_c),
        'station_supply_temp_warn_C': _workspace_series(station_supply_temp_warn_c),
        'station_return_temp_max_C': _workspace_series(station_return_temp_max_c),
        'station_cooling_feedforward_W': _workspace_series(station_cooling_feedforward_w),
        'station_cooling_capacity_W': _constant_workspace_series(station_cooling_capacity_w),
        'station_cooling_kp_W_per_K': _constant_workspace_series(STATION_COOLING_KP_W_PER_K),
    }

    for user_id, demand_kw in user_loads.items():
        opening = float(valve_settings.loc[user_id, 'valve_opening'])
        design_dt = float(valve_settings.loc[user_id].get('design_delta_t_c', DESIGN_DELTA_T_C))
        valve_opening = np.full(TOTAL_HOURS, opening)
        required_flow = demand_kw / (CP_WATER_KJ_PER_KG_K * design_dt)

        # 命名与 Simulink From Workspace 块保持一致：
        # user3_heat/user4_heat/user6_heat 接入 Controlled Heat Flow Source，单位 W。
        df[f'user{user_id}_heat'] = demand_kw * 1000.0
        df[f'user{user_id}_required_flow_kg_s'] = demand_kw / (CP_WATER_KJ_PER_KG_K * design_dt)
        df[f'user{user_id}_valve_opening'] = valve_opening

        mat_inputs[f'user{user_id}_heat'] = _workspace_series(demand_kw * 1000.0)
        mat_inputs[f'user{user_id}_required_flow_kg_s'] = _workspace_series(required_flow)
        mat_inputs[f'user{user_id}_valve_opening'] = _workspace_series(valve_opening)

    df.to_csv(SIMULINK_BOUNDARY_FILE, index=False)
    print(f"✅ 已导出固定阀门边界: {SIMULINK_BOUNDARY_FILE}")

    # 水泵仍可作为中心站控制量。这里保留变频压差曲线，阀门不参与小时级调节。
    P_MAX_PA = 451000.0
    P_MIN_PA = 200000.0
    max_load = max(float(np.max(flat_total_load)), 1.0)
    pump_pressure_seq = [
        max(P_MIN_PA, P_MAX_PA * (float(load_kw) / max_load) ** 2)
        for load_kw in flat_total_load
    ]
    pump_pressure_array = np.asarray(pump_pressure_seq, dtype=float)
    pump_pressure_workspace = _soft_start_series(pump_pressure_array)
    pd.DataFrame({
        'time_sec': pump_pressure_workspace[:, 0],
        'pump_pressure': pump_pressure_workspace[:, 1]
    }).to_csv('Simulink_30Days_Commands.csv', index=False)
    print("✅ 已导出中心站水泵压差指令: Simulink_30Days_Commands.csv")

    mat_inputs['pump_pressure'] = pump_pressure_workspace
    sio.savemat(SIMULINK_INPUT_MAT_FILE, mat_inputs)
    print(f"✅ 已导出与 Simulink From Workspace 同名的 MAT 输入: {SIMULINK_INPUT_MAT_FILE}")
    print(f"✅ 冷站供水温度设定: {STATION_SUPPLY_TEMP_SET_C:.1f} ℃ ({STATION_SUPPLY_TEMP_SET_K:.2f} K)")
    print(f"✅ 冷站供水报警阈值: {STATION_SUPPLY_TEMP_WARN_C:.1f} ℃")
    print(f"✅ 冷站回水上限阈值: {STATION_RETURN_TEMP_MAX_C:.1f} ℃")
    print(f"✅ 冷站温控比例系数: {STATION_COOLING_KP_W_PER_K:.2e} W/K")
    print(f"✅ 冷站最大制冷量建议上限: {station_cooling_capacity_w / 1000.0:.1f} kW")
    print(f"✅ 水泵压差软启动: 0 Pa -> {pump_pressure_array[0]:.0f} Pa，用时 {PUMP_PRESSURE_SOFT_START_SEC:.0f} s")


def step1_generate_blind_pressure_for_simulink():
    """兼容旧入口：现在同时生成用户需求、固定阀门开度和水泵指令。"""
    step1_generate_fixed_valve_boundaries_for_simulink()


def _mat_keys(mat_data):
    return [k for k in mat_data.keys() if not k.startswith('__')]


def _extract_struct_field(obj, field_name):
    obj = np.asarray(obj).squeeze()
    if getattr(obj, "dtype", None) is not None and obj.dtype.names and field_name in obj.dtype.names:
        return np.asarray(obj[field_name]).squeeze()
    if hasattr(obj, field_name):
        return np.asarray(getattr(obj, field_name)).squeeze()
    if obj.dtype == object:
        for item in obj.flat:
            found = _extract_struct_field(item, field_name)
            if found is not None:
                return found
    return None


def _find_mat_array(mat_data, names):
    for name in names:
        if name in mat_data:
            arr = np.asarray(mat_data[name]).squeeze()
            if arr.dtype.names is None and arr.size > 0:
                return arr.astype(float).flatten()
        if name.startswith('out.'):
            out_obj = mat_data.get('out')
            if out_obj is not None:
                arr = _extract_struct_field(out_obj, name.split('.', 1)[1])
                if arr is not None:
                    arr = np.asarray(arr).squeeze()
                    if arr.dtype.names is None and arr.size > 0:
                        return arr.astype(float).flatten()
    return None


def _hourly_average(values, time_sec, total_hours=TOTAL_HOURS):
    values = np.asarray(values, dtype=float).flatten()
    if values.size == total_hours:
        return values
    if time_sec is None or len(time_sec) != len(values):
        points_per_hour = max(len(values) // total_hours, 1)
        return np.array([
            np.nanmean(values[i * points_per_hour:(i + 1) * points_per_hour])
            for i in range(total_hours)
        ])

    time_sec = np.asarray(time_sec, dtype=float).flatten()
    hourly = []
    for hour in range(total_hours):
        start = hour * 3600.0
        end = (hour + 1) * 3600.0
        mask = (time_sec >= start) & (time_sec < end)
        if np.any(mask):
            hourly.append(np.nanmean(values[mask]))
        else:
            hourly.append(np.interp(start + 1800.0, time_sec, values))
    return np.asarray(hourly)


def _read_user_sim_result(mat_data, user_id, time_sec):
    flow = _find_mat_array(mat_data, [
        f'out.real_flow_{user_id}', f'out.flow_{user_id}', f'out.m_flow_{user_id}',
        f'real_flow_{user_id}', f'flow_{user_id}', f'm_flow_{user_id}',
        f'real_flow{user_id}', f'flow{user_id}'
    ])
    t_sup = _find_mat_array(mat_data, [
        f'out.T_sup_{user_id}', f'out.Tsup_{user_id}', f'out.T_supply_{user_id}',
        f'T_sup_{user_id}', f'Tsup_{user_id}', f'T_supply_{user_id}',
        f'T_sup{user_id}', f'Tsup{user_id}'
    ])
    t_ret = _find_mat_array(mat_data, [
        f'out.T_ret_{user_id}', f'out.Tret_{user_id}', f'out.T_return_{user_id}',
        f'T_ret_{user_id}', f'Tret_{user_id}', f'T_return_{user_id}',
        f'T_ret{user_id}', f'Tret{user_id}'
    ])
    q_sup = _find_mat_array(mat_data, [
        f'out.Q_sup_{user_id}', f'out.Q_served_{user_id}', f'out.Q_real_{user_id}',
        f'Q_sup_{user_id}', f'Q_served_{user_id}', f'Q_real_{user_id}',
        f'Q_actual_{user_id}', f'real_q_{user_id}', f'Q_sup{user_id}'
    ])

    result = {}
    if flow is not None:
        result['flow_kg_s'] = _hourly_average(flow, time_sec)
    if t_sup is not None:
        result['t_sup'] = _hourly_average(t_sup, time_sec)
    if t_ret is not None:
        result['t_ret'] = _hourly_average(t_ret, time_sec)
    if q_sup is not None:
        q_sup_hourly = _hourly_average(q_sup, time_sec)
        if np.nanmax(np.abs(q_sup_hourly)) > 100000:
            q_sup_hourly = q_sup_hourly / 1000.0
        result['q_sup_kw'] = np.abs(q_sup_hourly)

    if 'q_sup_kw' not in result and {'flow_kg_s', 't_sup', 't_ret'} <= result.keys():
        delta_t = result['t_ret'] - result['t_sup']
        result['q_sup_kw'] = np.maximum(result['flow_kg_s'] * CP_WATER_KJ_PER_KG_K * delta_t, 0.0)

    return result


def _recommend_opening(current_opening, required_flow, actual_flow, row):
    min_opening = float(row.get('min_opening', MIN_VALVE_OPENING))
    max_opening = float(row.get('max_opening', MAX_VALVE_OPENING))
    if not np.isfinite(actual_flow) or actual_flow <= 1e-6:
        raw = max_opening
    else:
        raw = current_opening * required_flow / actual_flow

    lower = max(min_opening, current_opening - VALVE_STEP_LIMIT)
    upper = min(max_opening, current_opening + VALVE_STEP_LIMIT)
    return float(np.clip(raw, lower, upper))


def _save_next_quarter_valve_settings(valve_settings, report):
    """按本次诊断结果生成下一季度阀门配置，只有调大/调小项会改开度。"""
    next_settings = valve_settings.reset_index().copy()
    for _, item in report.iterrows():
        if item['action'] not in ['调大', '调小']:
            continue
        user_id = int(item['user_id'])
        next_settings.loc[
            next_settings['user_id'] == user_id,
            'valve_opening'
        ] = round(float(item['suggested_opening']), 4)

    next_settings.to_csv(NEXT_VALVE_SETTING_FILE, index=False, encoding='utf-8-sig')
    return next_settings


def _write_station_side_diagnosis(report):
    report = report.copy()
    report['flow_adequacy_ratio'] = (
        report['actual_peak_flow_kg_s'] / report['required_peak_flow_kg_s']
    )

    all_non_valve = (report['action'] == '非阀门优先').all()
    min_flow_ratio = float(report['flow_adequacy_ratio'].min())
    max_unmet = report.loc[report['peak_unmet_kw'].idxmax()]
    low_delta_t = report['median_delta_t_c'] < DESIGN_DELTA_T_C * LOW_DELTA_T_RATIO

    lines = [
        '站侧诊断结论',
        '=' * 40,
        f"最大峰值缺冷: 用户{int(max_unmet['user_id'])}，{max_unmet['peak_unmet_kw']:.1f} kW",
        f"最大未满足率: {report['unmet_ratio'].max() * 100:.2f}%",
        f"最小高峰流量满足系数: {min_flow_ratio:.2f}",
        '',
        '各用户高峰流量满足系数 actual/required:',
    ]

    for _, item in report.iterrows():
        lines.append(
            f"  用户{int(item['user_id'])}: "
            f"{item['flow_adequacy_ratio']:.2f}, "
            f"中位ΔT={item['median_delta_t_c']:.2f}℃, "
            f"未满足率={item['unmet_ratio'] * 100:.2f}%"
        )

    lines.append('')
    if all_non_valve and min_flow_ratio >= 0.98:
        lines.extend([
            '判断: 当前缺冷不再是阀门开度或支路流量不足导致。',
            '建议: 保持现有阀门开度，不要继续调大阀门。',
        ])
        if low_delta_t.any():
            lines.extend([
                '同时检测到多数支路ΔT偏低，继续增加流量或提高泵压可能只会加重低温差运行。',
                '下一轮优先检查:',
                '1. 冷站供水温度设定与实际到户供水温度，建议先尝试降低供水设定0.5~1.0℃。',
                '2. 冷机/蓄冰实际输出是否覆盖用户需求、管网漏热和水泵热。',
                '3. 用户热源保护逻辑中的ΔT_max是否过低，或热源/温度测点是否存在时间错位。',
                '4. 若供水温度和冷源能力均正常，再检查泵压差；泵压差只在高峰流量不足时优先调整。',
            ])
        else:
            lines.extend([
                '下一轮优先检查冷源容量、供水温度设定和水泵压差设定。',
            ])
    else:
        lines.extend([
            '判断: 仍存在阀门或水力分配问题。',
            '建议: 先按 valve_adjustment_report.csv 调整标记为“调大/调小”的用户，再重新仿真。',
        ])

    with open(STATION_DIAGNOSIS_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    return lines


def step2_optimize_with_real_physics_data():
    """
    固定阀门逻辑下的二阶段分析：
    1. 经济调度仍以用户原始需求为约束；
    2. Simulink 结果只用于判断实际满足量、未满足量和下一季度阀门开度建议。
    """
    print("\n📌 固定阀门逻辑：经济调度仍采用原始用户需求，不再把 Simulink 反算冷量当作需求。")
    optimal_p_target = 1400
    current_ice_soc = 0.0
    total_cost = 0.0

    for day in range(DAYS_IN_MONTH):
        cost, pen, exc, next_ice = optimize_daily_dispatch(
            day, current_ice_soc, optimal_p_target, MONTHLY_LOAD[day]
        )
        total_cost += cost
        current_ice_soc = next_ice

    print(f"✅ 在 {optimal_p_target} kVA 限制下，按用户原始需求调度的月电度成本约为: {total_cost:.1f} 元")

    if not os.path.exists('sim_result.mat'):
        print("⚠️ 未找到 sim_result.mat。请先用 Simulink 读取固定阀门边界并保存运行结果，再执行本脚本生成阀门建议。")
        return

    print("\n📥 正在读取 Simulink 运行结果，评估各用户实际满足情况...")
    mat_data = sio.loadmat('sim_result.mat', squeeze_me=True, struct_as_record=False)
    time_sec = _find_mat_array(mat_data, ['tout', 'out.tout', 'time', 'out.time', 't'])
    valve_settings = load_or_create_valve_settings()
    report_rows = []
    missing_users = []

    for user_id, demand_kw in _user_load_map().items():
        sim = _read_user_sim_result(mat_data, user_id, time_sec)
        if 'q_sup_kw' not in sim:
            missing_users.append(user_id)
            continue

        q_sup_kw = sim['q_sup_kw']
        demand_kw = np.asarray(demand_kw, dtype=float)
        unmet_kw = np.maximum(demand_kw - q_sup_kw, 0.0)
        unmet_ratio = float(np.nansum(unmet_kw) / max(np.nansum(demand_kw), 1e-6))
        peak_unmet_kw = float(np.nanmax(unmet_kw))

        row = valve_settings.loc[user_id]
        current_opening = float(row['valve_opening'])
        design_dt = float(row.get('design_delta_t_c', DESIGN_DELTA_T_C))
        high_load_mask = demand_kw >= 0.80 * np.nanmax(demand_kw)
        required_flow = demand_kw / (CP_WATER_KJ_PER_KG_K * design_dt)
        required_peak_flow = float(np.nanpercentile(required_flow[high_load_mask], 90))

        actual_peak_flow = np.nan
        median_delta_t = np.nan
        if 'flow_kg_s' in sim:
            actual_peak_flow = float(np.nanpercentile(sim['flow_kg_s'][high_load_mask], 90))
        if {'t_sup', 't_ret'} <= sim.keys():
            median_delta_t = float(np.nanmedian(sim['t_ret'] - sim['t_sup']))

        new_opening = current_opening
        action = '保持'
        reason = '未满足率和温差均在允许范围内'

        if unmet_ratio > UNMET_RATIO_TOL:
            if np.isfinite(actual_peak_flow) and actual_peak_flow < required_peak_flow * 0.98:
                new_opening = _recommend_opening(current_opening, required_peak_flow * 1.05, actual_peak_flow, row)
                action = '调大'
                reason = '高峰期实际流量低于设计流量，固定阀门限制了最不利时段供冷'
            else:
                action = '非阀门优先'
                if np.isfinite(median_delta_t) and median_delta_t < design_dt * LOW_DELTA_T_RATIO:
                    reason = '流量不低且ΔT偏低，继续开阀或加泵压意义有限，优先检查供水温度、冷源输出和负荷施加逻辑'
                else:
                    reason = '流量基本够但仍缺冷，优先检查供水温度、冷源能力或泵压差'
        elif np.isfinite(median_delta_t) and median_delta_t < design_dt * LOW_DELTA_T_RATIO and np.isfinite(actual_peak_flow):
            new_opening = _recommend_opening(current_opening, required_peak_flow * 0.98, actual_peak_flow, row)
            if new_opening < current_opening:
                action = '调小'
                reason = '未缺冷但温差偏低，说明该支路可能过流，可适度关小以让水力分配给不利环路'

        report_rows.append({
            'user_id': user_id,
            'current_opening': current_opening,
            'suggested_opening': new_opening,
            'change_percent_point': (new_opening - current_opening) * 100.0,
            'action': action,
            'reason': reason,
            'unmet_ratio': unmet_ratio,
            'peak_unmet_kw': peak_unmet_kw,
            'required_peak_flow_kg_s': required_peak_flow,
            'actual_peak_flow_kg_s': actual_peak_flow,
            'flow_adequacy_ratio': actual_peak_flow / required_peak_flow if np.isfinite(actual_peak_flow) else np.nan,
            'median_delta_t_c': median_delta_t,
        })

    if missing_users:
        print("⚠️ 缺少以下用户的实际供冷信号，无法生成完整阀门建议:", missing_users)
        print("   请在 MATLAB 保存这些信号，例如 out.real_flow_3/T_sup_3/T_ret_3，或直接保存 real_flow_3。")
        print("   当前 mat 文件可见键:", _mat_keys(mat_data))

    if not report_rows:
        print("❌ 未读取到可用于诊断的用户供冷量。请把 Simulink 的用户实际流量和供回水温度保存为独立变量。")
        return

    report = pd.DataFrame(report_rows)
    report.to_csv(VALVE_REPORT_FILE, index=False, encoding='utf-8-sig')
    next_settings = _save_next_quarter_valve_settings(valve_settings, report)
    station_lines = _write_station_side_diagnosis(report)

    print("\n========== 下一季度阀门开度建议 ==========")
    for _, r in report.iterrows():
        print(
            f"用户{int(r['user_id'])}: {r['action']}，"
            f"{r['current_opening']:.2f} -> {r['suggested_opening']:.2f}，"
            f"未满足率 {r['unmet_ratio']*100:.2f}%，峰值缺冷 {r['peak_unmet_kw']:.1f} kW。"
        )
        print(f"  原因: {r['reason']}")
    print(f"✅ 详细报告已保存: {VALVE_REPORT_FILE}")
    print(f"✅ 下一季度阀门配置已保存: {NEXT_VALVE_SETTING_FILE}")
    print(next_settings[['user_id', 'valve_opening']].to_string(index=False))
    print(f"✅ 站侧诊断已保存: {STATION_DIAGNOSIS_FILE}")
    print("\n".join(station_lines[:8]))


def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else 'prepare'
    if mode in ['prepare', 'input']:
        step1_generate_fixed_valve_boundaries_for_simulink()
        print("\n下一步：请在 Simulink 中重新加载 Simulink_30Days_Input.mat 并运行模型。")
        print("仿真完成并保存 sim_result.mat 后，再执行: python monthcooling.py diagnose")
    elif mode in ['diagnose', 'report']:
        step2_optimize_with_real_physics_data()
    elif mode == 'all':
        step1_generate_fixed_valve_boundaries_for_simulink()
        print("\n⚠️ all 模式会立即读取现有 sim_result.mat；请确认它对应当前阀门配置。")
        step2_optimize_with_real_physics_data()
    else:
        print("用法: python monthcooling.py prepare | diagnose | all")


# 执行入口
if __name__ == "__main__":
    main()
