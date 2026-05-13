
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
DEFAULT_CONFIG_EXCEL_PATH = 'D:/minicondadaima/lianxi/中心站测试模板.xlsx'
DEFAULT_TYPICAL_DAY_LOAD_FILE = (
    'D:/study/和达能源站/zrs2026/峰值处理/已处理后/中心站秋季典型日.xlsx'
)
DEFAULT_REAL_MONTH_USER_LOAD_FILE = (
    'D:/minicondadaima/lianxi/duqudaochu/output/'
    '中心能源站-维亚园区+中心能源站-中心站地块+中心能源站-加速器五期高区+'
    '中心能源站-加速器五期低区+中心能源站-康洲园区_2025-10-01_00-00-00_'
    '2025-11-01_00-00-00_冷量汇总.xlsx'
)
DEFAULT_REAL_AUGUST_USER_LOAD_FILE = (
    'D:/minicondadaima/lianxi/duqudaochu/output/'
    '3号能源站-A线+3号能源站-B线+3号能源站-C线+3号能源站-D线+'
    '中心能源站-维亚园区+中心能源站-中心站地块+中心能源站-加速器五期高区+'
    '中心能源站-加速器五期低区+中心能源站-康洲园区_2025-08-01_00-00-00_'
    '2025-09-01_00-00-00_冷量汇总.xlsx'
)
DEFAULT_REAL_JANUARY_USER_LOAD_FILE = (
    'D:/minicondadaima/lianxi/duqudaochu/output/'
    '中心能源站-维亚园区+中心能源站-中心站地块+中心能源站-加速器五期高区+'
    '中心能源站-加速器五期低区+中心能源站-康洲园区_2026-01-01_00-00-00_'
    '2026-02-01_00-00-00_冷量汇总.xlsx'
)

USER_IDS = (1, 3, 4, 6)
USER_AREA_LABELS = {
    1: '三号站',
    3: '维亚园区',
    4: '加速器五区/五期',
    6: '康州/康洲园区',
}
USER_AREA_KEYWORDS = {
    1: ('三号站', '3号站', '3号能源站'),
    3: ('维亚园区',),
    4: ('加速器五区', '加速器五期'),
    6: ('康州园区', '康洲园区'),
}


def _cli_value(flag_name, default=None):
    if flag_name not in sys.argv:
        return default
    idx = sys.argv.index(flag_name)
    if idx + 1 >= len(sys.argv):
        return default
    return sys.argv[idx + 1]


def _cli_has(*tokens):
    return any(token in sys.argv for token in tokens)


def _read_external_load_profile(path):
    """
    读取外部典型日/负荷文件。
    支持:
      1. 原模板结构: sheet=负荷, column=load_kwc
      2. 典型日结构: 任意 sheet 中包含 总冷量/冷量/load_kwc 等列
    返回 24 点逐时 kW 冷负荷。
    """
    xl = pd.ExcelFile(path)
    candidate_sheets = ['负荷'] + [s for s in xl.sheet_names if s != '负荷']
    load_columns = ['load_kwc', '总冷量', '冷量', '总负荷', 'total_load', 'load_kw', 'load']

    for sheet in candidate_sheets:
        if sheet not in xl.sheet_names:
            continue
        df = pd.read_excel(path, sheet_name=sheet)
        matched_col = None
        for col in df.columns:
            col_text = str(col).strip()
            if col_text in load_columns:
                matched_col = col
                break
        if matched_col is None:
            continue

        values = pd.to_numeric(df[matched_col], errors='coerce').dropna().to_numpy(dtype=float)
        if values.size >= 25:
            values = values[:24]
        if values.size != HOURS_IN_DAY:
            raise ValueError(
                f"外部负荷文件 {path} 的 sheet={sheet} 列={matched_col} 有 {values.size} 个有效点，"
                f"当前典型日仿真需要 {HOURS_IN_DAY} 个逐时点。"
            )
        print(f"✅ 已读取外部典型日负荷: {path}")
        print(f"   sheet={sheet}, column={matched_col}, 峰值={np.max(values):.2f} kW, 均值={np.mean(values):.2f} kW")
        return values

    raise ValueError(f"外部负荷文件 {path} 中没有找到可识别的负荷列: {load_columns}")


def _read_real_user_month_profiles(path):
    """
    读取真实整月四个用户逐时冷负荷。
    映射关系:
      三号站 -> 用户1
      维亚园区 -> 用户3
      加速器五区/五期 -> 用户4
      康州/康洲园区 -> 用户6
    文件通常包含整月起点到次月1日0点的闭区间，末端点会自动裁掉。
    """
    area_col = '区域名称'
    time_col = '时间'
    load_col = '冷量汇总'

    df = pd.read_excel(path)
    missing_cols = [c for c in [area_col, time_col, load_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"真实用户负荷文件缺少列: {missing_cols}，实际列为: {list(df.columns)}")

    df = df[[area_col, time_col, load_col]].copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df[load_col] = pd.to_numeric(df[load_col], errors='coerce').fillna(0.0)

    start = df[time_col].min()
    end = df[time_col].max()
    if pd.isna(start) or pd.isna(end) or end <= start:
        raise ValueError("真实用户负荷文件时间范围无效。")
    if start.minute != 0 or start.second != 0 or end.minute != 0 or end.second != 0:
        raise ValueError(f"真实用户负荷文件不是整点逐时数据: start={start}, end={end}")
    df = df[(df[time_col] >= start) & (df[time_col] <= end)]

    def area_to_user_id(area_name):
        text = str(area_name).strip()
        for user_id in USER_IDS:
            if any(keyword in text for keyword in USER_AREA_KEYWORDS[user_id]):
                return user_id
        return np.nan

    df['_user_id'] = df[area_col].apply(area_to_user_id)
    selected = df[df['_user_id'].notna()].copy()
    selected['_user_id'] = selected['_user_id'].astype(int)
    pivot = (
        selected
        .pivot_table(index=time_col, columns='_user_id', values=load_col, aggfunc='sum')
        .sort_index()
    )

    expected_index_with_end = pd.date_range(start, end, freq='1h')
    pivot = pivot.reindex(expected_index_with_end)
    matched_user_ids = set(selected['_user_id'].unique())
    missing_existing = {
        f"用户{user_id}({USER_AREA_LABELS[user_id]})": int(pivot[user_id].isna().sum())
        for user_id in USER_IDS
        if user_id in matched_user_ids and user_id in pivot.columns and pivot[user_id].isna().any()
    }
    if missing_existing:
        raise ValueError(f"真实用户负荷文件存在缺失小时: {missing_existing}")

    missing_user_ids = [user_id for user_id in USER_IDS if user_id not in matched_user_ids]
    for user_id in missing_user_ids:
        pivot[user_id] = 0.0
    pivot = pivot[list(USER_IDS)]

    pivot = pivot[pivot.index < end]
    elapsed_hours = int(round((end - start).total_seconds() / 3600.0))
    if len(pivot) != elapsed_hours:
        raise ValueError(f"真实用户负荷应为 {elapsed_hours} 小时，当前为 {len(pivot)} 小时。")
    if elapsed_hours % HOURS_IN_DAY != 0:
        raise ValueError(f"真实用户负荷小时数 {elapsed_hours} 不能整除 24，请检查起止时间。")

    user_loads = {
        user_id: pivot[user_id].fillna(0.0).to_numpy(dtype=float)
        for user_id in USER_IDS
    }
    total = sum(user_loads.values())
    print(f"✅ 已读取真实整月四用户逐时负荷: {path}")
    print(f"   时间范围: {start} 到 {end}，仿真天数={elapsed_hours // HOURS_IN_DAY}，小时数={elapsed_hours}")
    if missing_user_ids:
        missing_names = ', '.join(f"用户{user_id}({USER_AREA_LABELS[user_id]})" for user_id in missing_user_ids)
        print(f"   ⚠️ 文件中未匹配到 {missing_names}，本次按 0 kW 处理。")
    for user_id in USER_IDS:
        values = user_loads[user_id]
        area_name = USER_AREA_LABELS[user_id]
        print(f"   用户{user_id}({area_name}): 峰值={np.max(values):.2f} kW, 均值={np.mean(values):.2f} kW")
    print(f"   四用户合计: 峰值={np.max(total):.2f} kW, 均值={np.mean(total):.2f} kW, 小时数={len(total)}")
    return user_loads


print("正在读取 Excel 设备与参数配置...")
excel_path = DEFAULT_CONFIG_EXCEL_PATH  # 设备参数与电价配置模板

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
external_load_file = _cli_value('--load-file')
real_user_load_file = _cli_value('--user-load-file')
if _cli_has('prepare_typical', 'diagnose_typical', 'typical'):
    external_load_file = external_load_file or DEFAULT_TYPICAL_DAY_LOAD_FILE
if _cli_has('prepare_real_january', 'diagnose_real_january', 'post_real_january', 'real_january'):
    real_user_load_file = real_user_load_file or DEFAULT_REAL_JANUARY_USER_LOAD_FILE
elif _cli_has('prepare_real_august', 'diagnose_real_august', 'post_real_august', 'real_august'):
    real_user_load_file = real_user_load_file or DEFAULT_REAL_AUGUST_USER_LOAD_FILE
elif _cli_has('prepare_real_month', 'diagnose_real_month', 'post_real_month', 'real_month'):
    real_user_load_file = real_user_load_file or DEFAULT_REAL_MONTH_USER_LOAD_FILE

real_user_loads = None
if real_user_load_file:
    real_user_loads = _read_real_user_month_profiles(real_user_load_file)
    BASE_LOAD = sum(real_user_loads.values())
    DAYS_IN_MONTH = int(len(BASE_LOAD) // HOURS_IN_DAY)
elif external_load_file:
    # 外部典型日只覆盖负荷曲线；设备、电价、经济参数仍来自中心站测试模板。
    BASE_LOAD = _read_external_load_profile(external_load_file)
    DAYS_IN_MONTH = int(_cli_value('--days', 1))
else:
    raise ValueError(
        "当前版本不再生成随机/比例模拟负荷。请使用真实用户热负荷文件，例如: "
        "python monthcoolingafter.py prepare_real_august，或 "
        "python monthcoolingafter.py prepare --user-load-file <真实用户冷量汇总.xlsx>"
    )

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
ECONOMIC_SCAN_FILE = 'monthly_economic_scan.csv'
MONTHLY_DISPATCH_FILE = 'monthly_dispatch_schedule.csv'
ECONOMIC_SUMMARY_FILE = 'monthly_economic_summary.txt'
# 没有四用户实测文件时，保持旧的三用户总负荷拆分不变；用户1默认为 0 kW。
# 四用户真实月度文件会覆盖这些比例。
USER_LOAD_RATIOS = {1: 0.0, 3: 1 / 22, 4: 9 / 22, 6: 12 / 22}
DIRECT_PIPE_USER_IDS = (1,)
VALVED_USER_IDS = tuple(user_id for user_id in USER_IDS if user_id not in DIRECT_PIPE_USER_IDS)
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
PUMP_PRESSURE_MIN_PA = 300000.0
PUMP_PRESSURE_MAX_PA = 1200000.0


# ==========================================
# 2. 生成 30 天逐时负荷并按比例切分 (供 Simulink 使用)
# ==========================================
print(f"正在生成 {DAYS_IN_MONTH} 天 ({TOTAL_HOURS}小时) 管网水力负荷边界条件...")
np.random.seed(42)
MONTHLY_LOAD = []
if real_user_loads is not None:
    print("正在使用真实四用户逐时负荷生成整月仿真边界，不再叠加随机天气扰动，也不再按比例拆分。")
    for day in range(DAYS_IN_MONTH):
        start_idx = day * HOURS_IN_DAY
        end_idx = (day + 1) * HOURS_IN_DAY
        MONTHLY_LOAD.append(BASE_LOAD[start_idx:end_idx])
elif external_load_file:
    print(f"正在使用外部典型日负荷生成 {DAYS_IN_MONTH} 天仿真边界，不再叠加随机天气扰动。")
    for day in range(DAYS_IN_MONTH):
        MONTHLY_LOAD.append(BASE_LOAD.copy())
else:
    for day in range(DAYS_IN_MONTH):
        # 模拟天气波动：每天在基准负荷上产生 ±10% 的随机浮动
        daily_variation = np.random.uniform(0.9, 1.1, size=HOURS_IN_DAY)
        MONTHLY_LOAD.append(BASE_LOAD * daily_variation)

# 展平为连续的 720 小时总负荷
flat_total_load = np.concatenate(MONTHLY_LOAD)

if real_user_loads is not None:
    load_by_user = {
        user_id: real_user_loads.get(user_id, np.zeros(TOTAL_HOURS, dtype=float))
        for user_id in USER_IDS
    }
else:
    # 旧模板只有总站总负荷时，按既有三用户比例拆分；新增用户1默认 0 kW。
    load_by_user = {
        user_id: flat_total_load * USER_LOAD_RATIOS.get(user_id, 0.0)
        for user_id in USER_IDS
    }

# 导出为 Simulink From Workspace 可直接读取的格式 [time_sec, value]
time_series_sec = np.arange(TOTAL_HOURS) * 3600  
simulink_load_data = {
    'time_sec': time_series_sec,
    'total_load': flat_total_load,
}
for user_id in USER_IDS:
    simulink_load_data[f'user{user_id}_load'] = load_by_user[user_id]
simulink_load_df = pd.DataFrame(simulink_load_data)
simulink_load_df.to_csv('Simulink_30Days_Load.csv', index=False)
print(" 物理边界已生成！文件已保存为 'Simulink_30Days_Load.csv'")

# ==========================================
# 3. 内环：单日日程调度优化器 (PuLP)
# ==========================================
def _safe_var_value(var, default=0.0):
    value = getattr(var, "varValue", None)
    if value is None:
        return default
    return float(value)


def optimize_daily_dispatch(day_idx, initial_ice, p_target_kva, daily_load, return_schedule=False):
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
    status_code = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus.get(status_code, str(status_code))
    
    cost_val = float(pulp.value(daily_energy_cost) or 0.0)
    penalty_val = float(pulp.value(penalty_cost) or 0.0)
    excess_val = sum([_safe_var_value(P_excess_kva[t]) for t in range(HOURS_IN_DAY)])
    next_soc = _safe_var_value(Tank_SOC[HOURS_IN_DAY])

    if not return_schedule:
        return cost_val, penalty_val, excess_val, next_soc, status

    rows = []
    for t in range(HOURS_IN_DAY):
        q1 = _safe_var_value(Q_chiller1[t])
        q2 = _safe_var_value(Q_chiller2[t])
        q_ice_discharge = _safe_var_value(Q_ice_discharge[t])
        q_ice_charge = _safe_var_value(Q_ice_charge[t])
        p_chiller_kw = q1 / CHILLER1_COP + q2 / CHILLER2_COP
        p_ice_make_kw = q_ice_charge / ICE_MAKE_COP
        p_pump_kw = 50 + 0.02 * float(daily_load[t])
        total_power_kw = p_chiller_kw + p_ice_make_kw + p_pump_kw
        total_power_kva = total_power_kw / PF
        rows.append({
            'day': day_idx + 1,
            'hour_in_day': t,
            'hour': day_idx * HOURS_IN_DAY + t,
            'sim_cooling_load_kw': float(daily_load[t]),
            'Q_chiller1_kw': q1,
            'Q_chiller2_kw': q2,
            'Q_ice_discharge_kw': q_ice_discharge,
            'Q_ice_charge_kw': q_ice_charge,
            'Tank_SOC_start_kWh': _safe_var_value(Tank_SOC[t]),
            'Tank_SOC_end_kWh': _safe_var_value(Tank_SOC[t + 1]),
            'P_chiller_kw': p_chiller_kw,
            'P_ice_make_kw': p_ice_make_kw,
            'P_pump_kw': p_pump_kw,
            'P_total_kw': total_power_kw,
            'P_total_kva': total_power_kva,
            'P_target_kva': float(p_target_kva),
            'P_excess_kva': _safe_var_value(P_excess_kva[t]),
            'tou_price': float(TOU_PRICES[t]),
            'hourly_energy_cost': total_power_kw * float(TOU_PRICES[t]),
            'solver_status': status,
        })
    
    return cost_val, penalty_val, excess_val, next_soc, status, rows

# ==========================================
# 4. 外环：需量扫描与全月博弈主程序
# ==========================================
def _build_p_targets(load_profile_kw):
    peak_load = max(float(np.nanmax(load_profile_kw)), 1.0)
    best_cop = max(float(CHILLER1_COP), float(CHILLER2_COP), 1.0)
    estimated_peak_kva = ((peak_load / best_cop) + 50.0 + 0.02 * peak_load) / PF
    lower = max(800, int(np.floor(estimated_peak_kva * 0.45 / 100.0) * 100))
    upper = max(lower + 100, int(np.ceil(estimated_peak_kva * 1.30 / 100.0) * 100))
    return range(lower, upper + 1, 100)


def scan_monthly_demand(load_profile_kw, source_label='sim_result'):
    load_profile_kw = np.asarray(load_profile_kw, dtype=float)
    usable_hours = (len(load_profile_kw) // HOURS_IN_DAY) * HOURS_IN_DAY
    if usable_hours <= 0:
        raise ValueError("经济调度负荷长度不足 24 小时，无法做日内调度。")
    if usable_hours != len(load_profile_kw):
        print(f"⚠️ 经济调度负荷长度 {len(load_profile_kw)} 不能整除24，已裁剪到 {usable_hours} 小时。")
        load_profile_kw = load_profile_kw[:usable_hours]

    days = usable_hours // HOURS_IN_DAY
    daily_loads = [
        load_profile_kw[day * HOURS_IN_DAY:(day + 1) * HOURS_IN_DAY]
        for day in range(days)
    ]
    p_targets_to_test = _build_p_targets(load_profile_kw)
    results = []
    
    print(f"\n🚀 启动外环容量阈值扫描 (基于 {source_label} 的 {days} 天仿真制冷量)...")
    print("-" * 96)
    print(f"{'P_target(kVA)':<14} | {'月电度电费(元)':<13} | {'容量费(元)':<12} | {'总成本(元)':<12} | {'罚款(元)':<12} | {'状态'}")
    print("-" * 96)
    
    for p_target in p_targets_to_test:
        monthly_energy_cost, monthly_penalty_cost, total_excess = 0.0, 0.0, 0.0
        current_ice_soc = 0.0
        statuses = []
        
        for day, daily_load in enumerate(daily_loads):
            cost, penalty, excess, next_ice, status = optimize_daily_dispatch(
                day, current_ice_soc, p_target, daily_load
            )
            monthly_energy_cost += cost
            monthly_penalty_cost += penalty
            total_excess += excess
            current_ice_soc = next_ice
            statuses.append(status)
            
        demand_charge = p_target * DEMAND_CHARGE_RATE
        total_cost = monthly_energy_cost + demand_charge
        is_feasible = total_excess < 1.0 and all(status == 'Optimal' for status in statuses)
        status_text = 'Optimal' if all(status == 'Optimal' for status in statuses) else '/'.join(sorted(set(statuses)))
        
        results.append({
            'P_target': p_target,
            'Energy_Cost': monthly_energy_cost,
            'Demand_Charge': demand_charge,
            'Total_Cost': total_cost,
            'Penalty': monthly_penalty_cost,
            'Total_Excess_kVA': total_excess,
            'Is_Feasible': is_feasible,
            'Solver_Status': status_text,
        })
        print(f"{p_target:<14} | {monthly_energy_cost:<13.1f} | {demand_charge:<12.1f} | {total_cost:<12.1f} | {monthly_penalty_cost:<12.1f} | {status_text}")

    scan_df = pd.DataFrame(results)
    scan_df.to_csv(ECONOMIC_SCAN_FILE, index=False, encoding='utf-8-sig')
    feasible_df = scan_df[scan_df['Is_Feasible'] == True].copy()
    
    if feasible_df.empty:
        print("\n❌ 警告：所有容量阈值均无法满足要求！请检查冷机/蓄冰能力、仿真制冷量峰值或放宽 KVA 扫描上限。")
        return scan_df, None, None
        
    optimal_row = feasible_df.loc[feasible_df['Total_Cost'].idxmin()]
    optimal_p = float(optimal_row['P_target'])

    current_ice_soc = 0.0
    dispatch_rows = []
    for day, daily_load in enumerate(daily_loads):
        cost, penalty, excess, next_ice, status, rows = optimize_daily_dispatch(
            day, current_ice_soc, optimal_p, daily_load, return_schedule=True
        )
        dispatch_rows.extend(rows)
        current_ice_soc = next_ice
    dispatch_df = pd.DataFrame(dispatch_rows)
    dispatch_df.to_csv(MONTHLY_DISPATCH_FILE, index=False, encoding='utf-8-sig')
    
    print("-" * 96)
    print(f"✅ 寻优完成！最优月度报装需量为: ** {optimal_p:.0f} kVA **")
    print(f"预计月度总账单为: {optimal_row['Total_Cost']:.1f} 元")
    print(f"✅ 外环扫描已保存: {ECONOMIC_SCAN_FILE}")
    print(f"✅ 每月逐时调度已保存: {MONTHLY_DISPATCH_FILE}")
    print("-" * 96)

    plt.figure(figsize=(10, 6))
    plt.plot(feasible_df['P_target'], feasible_df['Total_Cost'], marker='o', label='Total Cost (总账单)', color='purple')
    plt.plot(feasible_df['P_target'], feasible_df['Energy_Cost'], linestyle='--', label='Energy Cost (谷峰电度费)', color='blue')
    plt.plot(feasible_df['P_target'], feasible_df['Demand_Charge'], linestyle='--', label='Demand Charge (容量费)', color='orange')
    plt.axvline(x=optimal_p, color='red', linestyle='-.', label=f'Optimal: {optimal_p:.0f} kVA')
    plt.title('Capacity Demand (kVA) vs. Monthly Cost Optimization')
    plt.xlabel('Target Maximum Apparent Power (kVA)')
    plt.ylabel('Monthly Cost (RMB)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('optimization_result.png')
    plt.close()

    summary_lines = [
        '月度经济调度汇总',
        '=' * 40,
        f'负荷来源: {source_label}',
        f'调度小时数: {usable_hours}',
        f'调度天数: {days}',
        f'仿真制冷量峰值: {np.nanmax(load_profile_kw):.2f} kW',
        f'仿真制冷量均值: {np.nanmean(load_profile_kw):.2f} kW',
        f'最优报装需量: {optimal_p:.0f} kVA',
        f"月电度电费: {optimal_row['Energy_Cost']:.1f} 元",
        f"容量费: {optimal_row['Demand_Charge']:.1f} 元",
        f"总成本: {optimal_row['Total_Cost']:.1f} 元",
        f'外环扫描文件: {ECONOMIC_SCAN_FILE}',
        f'逐时调度文件: {MONTHLY_DISPATCH_FILE}',
    ]
    with open(ECONOMIC_SUMMARY_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines) + '\n')
    print(f"✅ 经济性汇总已保存: {ECONOMIC_SUMMARY_FILE}")
    return scan_df, dispatch_df, optimal_row

# ==========================================
# 5. 固定管径 + 季度固定阀门的 Simulink 边界与结果诊断
# ==========================================
import os
import scipy.io as sio # 用于读取 Matlab 数据


def _user_load_map():
    return {
        user_id: load_by_user[user_id]
        for user_id in USER_IDS
    }


def load_or_create_valve_settings():
    """
    读取季度阀门开度。如果文件不存在，则按默认开度创建模板。
    用户1已改为直通管，不再出现在阀门配置中。
    valve_opening 使用 0~1 表示，Simulink 中可映射为阀门 Kv 或局部阻力系数。
    """
    def default_row(user_id):
        demand_kw = np.asarray(_user_load_map()[user_id], dtype=float)
        design_load_kw = float(np.nanmax(demand_kw)) if demand_kw.size else 0.0
        design_flow_kg_s = design_load_kw / (CP_WATER_KJ_PER_KG_K * DESIGN_DELTA_T_C)
        return {
            'user_id': user_id,
            'valve_opening': DEFAULT_VALVE_OPENING[user_id],
            'min_opening': MIN_VALVE_OPENING,
            'max_opening': MAX_VALVE_OPENING,
            'design_delta_t_c': DESIGN_DELTA_T_C,
            'design_flow_kg_s': design_flow_kg_s,
        }

    if os.path.exists(VALVE_SETTING_FILE):
        settings = pd.read_csv(VALVE_SETTING_FILE)
        changed = False
    else:
        settings = pd.DataFrame([default_row(user_id) for user_id in VALVED_USER_IDS])
        settings.to_csv(VALVE_SETTING_FILE, index=False, encoding='utf-8-sig')
        print(f"✅ 已创建季度阀门开度模板: {VALVE_SETTING_FILE}，请按现场实际开度修正后再跑 Simulink。")
        changed = False

    settings['user_id'] = settings['user_id'].astype(int)
    required_columns = {
        'valve_opening': np.nan,
        'min_opening': MIN_VALVE_OPENING,
        'max_opening': MAX_VALVE_OPENING,
        'design_delta_t_c': DESIGN_DELTA_T_C,
        'design_flow_kg_s': np.nan,
    }
    for column, default_value in required_columns.items():
        if column not in settings.columns:
            settings[column] = default_value
            changed = True

    existing_user_ids = set(settings['user_id'].astype(int))
    missing_user_ids = [user_id for user_id in VALVED_USER_IDS if user_id not in existing_user_ids]
    if missing_user_ids:
        settings = pd.concat(
            [settings, pd.DataFrame([default_row(user_id) for user_id in missing_user_ids])],
            ignore_index=True,
        )
        changed = True
        missing_names = ', '.join(f"用户{user_id}" for user_id in missing_user_ids)
        print(f"⚠️ {VALVE_SETTING_FILE} 缺少 {missing_names}，已按默认开度补齐。")

    removed_direct_users = sorted(existing_user_ids.intersection(DIRECT_PIPE_USER_IDS))
    settings = settings[settings['user_id'].isin(VALVED_USER_IDS)].copy()
    if removed_direct_users:
        changed = True
        removed_names = ', '.join(f"用户{user_id}" for user_id in removed_direct_users)
        print(f"ℹ️ {removed_names} 已改为直通管，已从 {VALVE_SETTING_FILE} 中移除。")

    for user_id in VALVED_USER_IDS:
        mask = settings['user_id'] == user_id
        row = default_row(user_id)
        for column in required_columns:
            current = pd.to_numeric(settings.loc[mask, column], errors='coerce')
            if current.isna().any():
                settings.loc[mask, column] = row[column]
                changed = True
        design_flow = pd.to_numeric(settings.loc[mask, 'design_flow_kg_s'], errors='coerce')
        if (design_flow.fillna(0.0) <= 1e-9).any() and row['design_flow_kg_s'] > 1e-9:
            settings.loc[mask, 'design_flow_kg_s'] = row['design_flow_kg_s']
            changed = True

    order = {user_id: idx for idx, user_id in enumerate(VALVED_USER_IDS)}
    settings['_order'] = settings['user_id'].map(order)
    settings = settings.sort_values('_order').drop(columns=['_order'])
    if changed:
        settings.to_csv(VALVE_SETTING_FILE, index=False, encoding='utf-8-sig')
        print(f"✅ 已更新季度阀门开度模板为三路可调阀格式: {VALVE_SETTING_FILE}")

    return settings.set_index('user_id')


def _workspace_series(values):
    """From Workspace 使用的 [time_sec, value] 矩阵。"""
    values = np.asarray(values, dtype=float)
    times = time_series_sec
    terminal_time = float(TOTAL_HOURS * 3600)
    if values.size == TOTAL_HOURS:
        times = np.concatenate((time_series_sec, [terminal_time]))
        values = np.concatenate((values, [values[-1]]))
    return np.column_stack((times, values))


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

    times = np.concatenate(([0.0, float(ramp_sec)], time_series_sec[1:], [float(TOTAL_HOURS * 3600)]))
    values = np.concatenate(([float(start_value), hourly_values[0]], hourly_values[1:], [hourly_values[-1]]))
    return _workspace_series_with_time(times, values)


def step1_generate_fixed_valve_boundaries_for_simulink():
    """
    生成供 Simulink 使用的长期边界条件。
    用户负荷是外生需求；用户1为直通管，用户3/4/6阀门开度是季度固定参数。
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
        if user_id in VALVED_USER_IDS:
            opening = float(valve_settings.loc[user_id, 'valve_opening'])
            design_dt = float(valve_settings.loc[user_id].get('design_delta_t_c', DESIGN_DELTA_T_C))
            valve_opening = np.full(TOTAL_HOURS, opening)
        else:
            design_dt = DESIGN_DELTA_T_C
            valve_opening = None
        required_flow = demand_kw / (CP_WATER_KJ_PER_KG_K * design_dt)

        # 命名与 Simulink From Workspace 块保持一致：
        # user1_heat/user3_heat/user4_heat/user6_heat 接入 Controlled Heat Flow Source，单位 W。
        # 用户1为直通管，不再导出 user1_valve_opening。
        df[f'user{user_id}_heat'] = demand_kw * 1000.0
        df[f'user{user_id}_required_flow_kg_s'] = demand_kw / (CP_WATER_KJ_PER_KG_K * design_dt)

        mat_inputs[f'user{user_id}_heat'] = _workspace_series(demand_kw * 1000.0)
        mat_inputs[f'user{user_id}_required_flow_kg_s'] = _workspace_series(required_flow)
        if valve_opening is not None:
            df[f'user{user_id}_valve_opening'] = valve_opening
            mat_inputs[f'user{user_id}_valve_opening'] = _workspace_series(valve_opening)

    df.to_csv(SIMULINK_BOUNDARY_FILE, index=False)
    print(f"✅ 已导出用户边界: {SIMULINK_BOUNDARY_FILE}")
    print("✅ 用户1按直通管处理；用户3/4/6保留季度固定阀门开度。")

    # 水泵仍可作为中心站控制量。这里保留变频压差曲线，阀门不参与小时级调节。
    max_load = max(float(np.max(flat_total_load)), 1.0)
    pump_pressure_seq = [
        max(PUMP_PRESSURE_MIN_PA, PUMP_PRESSURE_MAX_PA * (float(load_kw) / max_load) ** 2)
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
    print(f"✅ 水泵压差范围: {PUMP_PRESSURE_MIN_PA:.0f} Pa ~ {PUMP_PRESSURE_MAX_PA:.0f} Pa")
    print(f"✅ 水泵压差软启动: 0 Pa -> {pump_pressure_array[0]:.0f} Pa，用时 {PUMP_PRESSURE_SOFT_START_SEC:.0f} s")


def step1_generate_blind_pressure_for_simulink():
    """兼容旧入口：现在同时生成用户需求、可调支路固定阀门开度和水泵指令。"""
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


def _read_user_sim_result(mat_data, user_id, time_sec, total_hours=TOTAL_HOURS):
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
        result['flow_kg_s'] = _hourly_average(flow, time_sec, total_hours=total_hours)
    if t_sup is not None:
        result['t_sup'] = _hourly_average(t_sup, time_sec, total_hours=total_hours)
    if t_ret is not None:
        result['t_ret'] = _hourly_average(t_ret, time_sec, total_hours=total_hours)
    if q_sup is not None:
        q_sup_hourly = _hourly_average(q_sup, time_sec, total_hours=total_hours)
        if np.nanmax(np.abs(q_sup_hourly)) > 100000:
            q_sup_hourly = q_sup_hourly / 1000.0
        result['q_sup_kw'] = np.abs(q_sup_hourly)

    if 'q_sup_kw' not in result and {'flow_kg_s', 't_sup', 't_ret'} <= result.keys():
        delta_t = result['t_ret'] - result['t_sup']
        result['q_sup_kw'] = np.maximum(result['flow_kg_s'] * CP_WATER_KJ_PER_KG_K * delta_t, 0.0)

    return result


def _sim_total_hours_from_time(time_sec):
    if time_sec is None or len(time_sec) == 0:
        return TOTAL_HOURS
    return max(int(np.floor(np.nanmax(np.asarray(time_sec, dtype=float)) / 3600.0)), 1)


def _read_simulated_cooling_load_kw(mat_data, time_sec, total_hours, user_sim_results=None):
    station_cooling, station_name = None, None
    for names in [
        ['Q_station_cooling', 'out.Q_station_cooling'],
        ['Q_cooling_removed', 'out.Q_cooling_removed'],
        ['Q_cool', 'out.Q_cool'],
        ['station_cooling_W', 'out.station_cooling_W'],
    ]:
        station_cooling = _find_mat_array(mat_data, names)
        if station_cooling is not None:
            station_name = names[0]
            break

    if station_cooling is not None:
        cooling_kw = _hourly_average(station_cooling, time_sec, total_hours=total_hours)
        if np.nanmax(np.abs(cooling_kw)) > 100000:
            cooling_kw = cooling_kw / 1000.0
        cooling_kw = np.abs(cooling_kw)
        if np.nanmax(cooling_kw) > 1e-6:
            return cooling_kw, station_name

    if user_sim_results is None:
        user_sim_results = {
            user_id: _read_user_sim_result(mat_data, user_id, time_sec, total_hours=total_hours)
            for user_id in USER_IDS
        }
    q_cols = [
        np.asarray(sim['q_sup_kw'], dtype=float)
        for sim in user_sim_results.values()
        if 'q_sup_kw' in sim
    ]
    if not q_cols:
        raise ValueError("sim_result.mat 中没有 Q_station_cooling，也无法由用户流量和供回水温度计算实际供冷量。")
    return np.sum(q_cols, axis=0), 'sum_user_actual_cooling'


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
    report['flow_adequacy_ratio'] = np.where(
        report['required_peak_flow_kg_s'] > 1e-9,
        report['actual_peak_flow_kg_s'] / report['required_peak_flow_kg_s'],
        np.nan,
    )

    has_valve_adjustment = report['action'].isin(['调大', '调小']).any()
    has_direct_pipe_flow_issue = (report['action'] == '直通管流量不足').any()
    valid_flow_ratios = report['flow_adequacy_ratio'].replace([np.inf, -np.inf], np.nan).dropna()
    min_flow_ratio = float(valid_flow_ratios.min()) if not valid_flow_ratios.empty else np.nan
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
    if not has_valve_adjustment and not has_direct_pipe_flow_issue and min_flow_ratio >= 0.98:
        lines.extend([
            '判断: 当前缺冷不优先由可调阀门开度或支路流量不足导致。',
            '建议: 保持用户3/4/6现有阀门开度；用户1为直通管，无阀门开度可调。',
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
    elif has_direct_pipe_flow_issue and not has_valve_adjustment:
        lines.extend([
            '判断: 用户1为直通管，无阀门开度可调；当前问题更可能来自直通支路水力阻力、管径或泵压分配。',
            '建议: 优先检查用户1支路管径、局部阻力、连接方向、旁通/止回设置和总泵压差，再判断是否需要调整用户3/4/6阀门。',
        ])
    else:
        lines.extend([
            '判断: 仍存在阀门或水力分配问题。',
            '建议: 先按 valve_adjustment_report.csv 调整标记为“调大/调小”的用户3/4/6，再重新仿真。',
        ])

    with open(STATION_DIAGNOSIS_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    return lines


def step2_optimize_with_real_physics_data():
    """
    固定阀门逻辑下的二阶段分析：
    1. 先读取 Simulink 实际制冷量，并据此做外环需量扫描和内环逐时调度；
    2. Simulink 结果同时用于判断实际满足量、未满足量和下一季度阀门开度建议。
    """
    print("\n📌 仿真后分析：经济调度采用 Simulink 实际制冷量；阀门建议仍以真实用户需求和实际供冷差额判断。")

    if not os.path.exists('sim_result.mat'):
        print("⚠️ 未找到 sim_result.mat。请先用 Simulink 读取固定阀门边界并保存运行结果，再执行本脚本生成阀门建议。")
        return

    print("\n📥 正在读取 Simulink 运行结果，评估各用户实际满足情况...")
    mat_data = sio.loadmat('sim_result.mat', squeeze_me=True, struct_as_record=False)
    time_sec = _find_mat_array(mat_data, ['tout', 'out.tout', 'time', 'out.time', 't'])
    sim_total_hours = min(_sim_total_hours_from_time(time_sec), TOTAL_HOURS)
    print(f"✅ 检测到仿真结果时长: {sim_total_hours} 小时 ({sim_total_hours / 24.0:.2f} 天)")

    user_sim_results = {
        user_id: _read_user_sim_result(mat_data, user_id, time_sec, total_hours=sim_total_hours)
        for user_id in USER_IDS
    }
    try:
        sim_cooling_load_kw, cooling_source = _read_simulated_cooling_load_kw(
            mat_data, time_sec, sim_total_hours, user_sim_results=user_sim_results
        )
        scan_monthly_demand(sim_cooling_load_kw, source_label=cooling_source)
    except Exception as exc:
        print(f"⚠️ 经济调度未完成: {exc}")

    valve_settings = load_or_create_valve_settings()
    report_rows = []
    missing_users = []

    for user_id, demand_kw in _user_load_map().items():
        sim = user_sim_results[user_id]
        if 'q_sup_kw' not in sim:
            missing_users.append(user_id)
            continue

        q_sup_kw = sim['q_sup_kw']
        demand_kw = np.asarray(demand_kw, dtype=float)[:sim_total_hours]
        q_sup_kw = np.asarray(q_sup_kw, dtype=float)[:sim_total_hours]
        unmet_kw = np.maximum(demand_kw - q_sup_kw, 0.0)
        unmet_ratio = float(np.nansum(unmet_kw) / max(np.nansum(demand_kw), 1e-6))
        peak_unmet_kw = float(np.nanmax(unmet_kw))

        is_valved_user = user_id in VALVED_USER_IDS
        if is_valved_user:
            row = valve_settings.loc[user_id]
            current_opening = float(row['valve_opening'])
            design_dt = float(row.get('design_delta_t_c', DESIGN_DELTA_T_C))
        else:
            row = None
            current_opening = np.nan
            design_dt = DESIGN_DELTA_T_C
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
        if is_valved_user:
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
        else:
            action = '无阀直通'
            reason = '用户1为直通管，无阀门开度可调；本行仅用于观察流量、温差和缺冷情况'
            if unmet_ratio > UNMET_RATIO_TOL:
                if np.isfinite(actual_peak_flow) and actual_peak_flow < required_peak_flow * 0.98:
                    action = '直通管流量不足'
                    reason = '用户1无阀可调且高峰实际流量低于需求流量，优先检查该支路管径、局部阻力、连接方向和总泵压差'
                else:
                    action = '非阀门优先'
                    reason = '用户1无阀且流量基本够但仍缺冷，优先检查冷站供水温度、冷源能力和热源施加逻辑'

        flow_adequacy_ratio = (
            actual_peak_flow / required_peak_flow
            if np.isfinite(actual_peak_flow) and required_peak_flow > 1e-9
            else np.nan
        )
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
            'flow_adequacy_ratio': flow_adequacy_ratio,
            'median_delta_t_c': median_delta_t,
        })

    if missing_users:
        print("⚠️ 缺少以下用户的实际供冷信号，无法生成完整阀门建议:", missing_users)
        print("   请在 MATLAB 保存这些信号，例如 out.real_flow_1/T_sup_1/T_ret_1，或直接保存 real_flow_1。")
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
        current_opening = float(r['current_opening'])
        suggested_opening = float(r['suggested_opening'])
        if np.isfinite(current_opening) and np.isfinite(suggested_opening):
            opening_text = f"{current_opening:.2f} -> {suggested_opening:.2f}"
        else:
            opening_text = "无阀门开度"
        print(
            f"用户{int(r['user_id'])}: {r['action']}，"
            f"{opening_text}，"
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
    if mode in [
        'prepare', 'input', 'prepare_typical', 'typical',
        'prepare_real_month', 'real_month', 'prepare_real_august', 'real_august',
        'prepare_real_january', 'real_january'
    ]:
        step1_generate_fixed_valve_boundaries_for_simulink()
        print("\n下一步：请在 Simulink 中重新加载 Simulink_30Days_Input.mat 并运行模型。")
        print("仿真完成并保存 sim_result.mat 后，再执行: python monthcoolingafter.py diagnose")
    elif mode in [
        'diagnose', 'post', 'report', 'diagnose_typical',
        'diagnose_real_month', 'diagnose_real_august', 'diagnose_real_january',
        'post_real_month', 'post_real_august', 'post_real_january'
    ]:
        step2_optimize_with_real_physics_data()
    elif mode == 'all':
        step1_generate_fixed_valve_boundaries_for_simulink()
        print("\n⚠️ all 模式会立即读取现有 sim_result.mat；请确认它对应当前阀门配置。")
        step2_optimize_with_real_physics_data()
    else:
        print("用法: python monthcoolingafter.py prepare_real_month | prepare_real_august | prepare_real_january | post_real_month | post_real_august | post_real_january | all")
        print("可选: --user-load-file <真实用户整月冷量汇总.xlsx>")
        print("可选: --user-load-file <真实四用户整月负荷.xlsx>")


# 执行入口
if __name__ == "__main__":
    main()
