# pip install pulp pandas numpy

import pulp
import numpy as np
import pandas as pd

# =========================
# 1) 输入数据
# =========================
hours = list(range(24))
dt = 1.0  # 小时

# 典型日需冷量(kW冷) - 按你提供数据
Q_load = {
    0: 6759.02, 1: 6833.49, 2: 6756.07, 3: 6761.13, 4: 6735.04, 5: 6782.21,
    6: 6736.83, 7: 6907.68, 8: 7278.19, 9: 8062.28, 10: 8739.60, 11: 8618.55,
    12: 8602.05, 13: 8499.88, 14: 8447.85, 15: 7561.75, 16: 7335.91, 17: 7369.50,
    18: 7425.73, 19: 7146.04, 20: 7373.37, 21: 7078.45, 22: 7373.73, 23: 7136.02
}

# 分时电价(元/kWh)
price = {}
for h in hours:
    if 0 <= h < 8:
        price[h] = 0.2203                 # 谷
    elif h in [9, 10, 15, 16]:
        price[h] = 1.1477                 # 尖
    elif h in [8, 11, 12, 17, 18, 19, 20, 21, 22]:
        price[h] = 0.9564                 # 峰
    elif h in [13, 14, 23]:
        price[h] = 0.5796                 # 平
    else:
        # 若有未覆盖小时，按平价
        price[h] = 0.5796

# 电制冷机参数
# 每台：设计冷量(kW冷) + COP在25/50/75/100%
chillers = {
    "C1": {"Qmax": 7032, "cop_25": 4.388, "cop_50": 5.167, "cop_75": 5.583, "cop_100": 5.599},
    "C2": {"Qmax": 3516, "cop_25": 4.695, "cop_50": 5.543, "cop_75": 5.760, "cop_100": 5.390},
    "C3": {"Qmax": 3517, "cop_25": 6.620, "cop_50": 7.520, "cop_75": 7.430, "cop_100": 6.260},
    "C4": {"Qmax": 2210, "cop_25": 3.330, "cop_50": 4.720, "cop_75": 5.580, "cop_100": 6.020},
}

# 蓄冰参数
Q_ice_charge_max = 5016.0       # 蓄冰机制冷量上限(kW冷)
P_ice_charge_max = 1182.0       # 蓄冰机电功率上限(kW)
COP_ice_charge = 4.24

RT_to_kW = 3.517
E_ice_max = 21000 * RT_to_kW    # kWh冷
Q_discharge_max = 0.13 * E_ice_max  # kW冷（按题意）

# 计费参数
pf = 0.85
demand_price = 48.0             # 元/kVA·月
days_in_month = 30              # 用典型日折算
daily_demand_coeff = demand_price / pf  # 元/(kW·月)
# 折算到“日目��函数”的需量成本（把月费均摊到典型日）
daily_demand_cost_factor = daily_demand_coeff / days_in_month  # 元/(kW·日)

# 非制冷其他基础负荷(如有可填，没数据先置0)
P_base = {h: 0.0 for h in hours}

# 可选：需量红线(kW)，None表示不强制
P_cap = None   # 例如 2500

# =========================
# 2) 分段线性化设置
# =========================
# 用4个负荷段：0-25,25-50,50-75,75-100%Qmax
segments = [
    ("s1", 0.25, None),  # COP用25%点
    ("s2", 0.25, None),  # COP用50%点
    ("s3", 0.25, None),  # COP用75%点
    ("s4", 0.25, None),  # COP用100%点
]

# 给每台机组补上各段COP（段内常数近似）
for c, d in chillers.items():
    d["seg_cop"] = [d["cop_25"], d["cop_50"], d["cop_75"], d["cop_100"]]

# =========================
# 3) 建模
# =========================
m = pulp.LpProblem("IceStorage_TOU_Demand_Optimization", pulp.LpMinimize)

# 决策变量
q_ch = {(c, h): pulp.LpVariable(f"q_{c}_{h}", lowBound=0) for c in chillers for h in hours}  # 各机组制冷量
p_ch = {(c, h): pulp.LpVariable(f"p_{c}_{h}", lowBound=0) for c in chillers for h in hours}  # 各机组电功率

q_seg = {(c, h, s): pulp.LpVariable(f"qseg_{c}_{h}_{s}", lowBound=0)
         for c in chillers for h in hours for s,_,_ in segments}

q_ice_ch = {h: pulp.LpVariable(f"q_ice_ch_{h}", lowBound=0, upBound=Q_ice_charge_max) for h in hours}  # 蓄冰机产冷(入冰罐)
p_ice_ch = {h: pulp.LpVariable(f"p_ice_ch_{h}", lowBound=0, upBound=P_ice_charge_max) for h in hours}  # 蓄冰机电功率

q_dis = {h: pulp.LpVariable(f"q_dis_{h}", lowBound=0, upBound=Q_discharge_max) for h in hours}  # 放冷功率
soc = {h: pulp.LpVariable(f"soc_{h}", lowBound=0, upBound=E_ice_max) for h in hours}            # 冰罐SOC(kWh冷)

p_total = {h: pulp.LpVariable(f"p_total_{h}", lowBound=0) for h in hours}
p_peak = pulp.LpVariable("p_peak", lowBound=0)  # 当日峰值功率(kW)

# 约束：机组分段
for c, d in chillers.items():
    Qmax = d["Qmax"]
    seg_caps = [0.25*Qmax, 0.25*Qmax, 0.25*Qmax, 0.25*Qmax]

    for h in hours:
        # 总制冷量 = 分段和
        m += q_ch[(c,h)] == pulp.lpSum(q_seg[(c,h,s)] for s,_,_ in segments)

        # 各段上限
        for idx, (s,_,_) in enumerate(segments):
            m += q_seg[(c,h,s)] <= seg_caps[idx]

        # 电功率 = 分段制冷量 / 对应段COP
        m += p_ch[(c,h)] == pulp.lpSum(q_seg[(c,h,s)] / d["seg_cop"][idx]
                                       for idx, (s,_,_) in enumerate(segments))

# 蓄冰机COP关系
for h in hours:
    m += q_ice_ch[h] == COP_ice_charge * p_ice_ch[h]

# 冷量平衡：电制冷 + 放冷 = 负荷 + 充冰（充冰相当于额外“冷量需求”）
for h in hours:
    m += pulp.lpSum(q_ch[(c,h)] for c in chillers) + q_dis[h] == Q_load[h] + q_ice_ch[h]

# SOC动态
# 设日初SOC = 日末SOC（周期约束），避免“透支/囤积”造成偏差
for h in hours:
    if h == 0:
        m += soc[h] == soc[23] + q_ice_ch[h]*dt - q_dis[h]*dt
    else:
        m += soc[h] == soc[h-1] + q_ice_ch[h]*dt - q_dis[h]*dt

# 总功率与峰值
for h in hours:
    m += p_total[h] == P_base[h] + pulp.lpSum(p_ch[(c,h)] for c in chillers) + p_ice_ch[h]
    m += p_peak >= p_total[h]
    if P_cap is not None:
        m += p_total[h] <= P_cap

# 目标函数：典型日分时电费 + 典型日对应需量成本
energy_cost_day = pulp.lpSum(price[h] * p_total[h] * dt for h in hours)
demand_cost_day = daily_demand_cost_factor * p_peak
m += energy_cost_day + demand_cost_day

# =========================
# 4) 求解
# =========================
solver = pulp.PULP_CBC_CMD(msg=False)
m.solve(solver)

status = pulp.LpStatus[m.status]
print("Status:", status)

# =========================
# 5) 结果整理
# =========================
res = []
for h in hours:
    row = {
        "hour": h,
        "Q_load_kWc": Q_load[h],
        "Q_dis_kWc": q_dis[h].value(),
        "Q_ice_ch_kWc": q_ice_ch[h].value(),
        "P_ice_ch_kW": p_ice_ch[h].value(),
        "SOC_kWhc": soc[h].value(),
        "P_total_kW": p_total[h].value(),
        "price": price[h],
        "cost_hour_yuan": price[h] * p_total[h].value() * dt
    }
    for c in chillers:
        row[f"Q_{c}_kWc"] = q_ch[(c,h)].value()
        row[f"P_{c}_kW"] = p_ch[(c,h)].value()
    res.append(row)

df = pd.DataFrame(res)

energy_day = df["cost_hour_yuan"].sum()
peak_kw = p_peak.value()
demand_day = daily_demand_cost_factor * peak_kw
total_day = energy_day + demand_day

print(f"Energy cost/day: {energy_day:,.2f} 元")
print(f"Peak kW (day):   {peak_kw:,.2f} kW")
print(f"Demand cost/day: {demand_day:,.2f} 元 (按30天折算)")
print(f"Total cost/day:  {total_day:,.2f} 元")

# 按月估算
energy_month = energy_day * days_in_month
demand_month = (demand_price/pf) * peak_kw
total_month = energy_month + demand_month

print("\n--- Monthly Estimation ---")
print(f"Energy cost/month: {energy_month:,.2f} 元")
print(f"Demand cost/month: {demand_month:,.2f} 元")
print(f"Total cost/month:  {total_month:,.2f} 元")

# 导出
df.to_excel("dispatch_result.xlsx", index=False)
print("\nSaved: dispatch_result.xlsx")