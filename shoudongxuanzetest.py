import pulp
import numpy as np
import pandas as pd
import matlab.engine
import matplotlib.pyplot as plt

# =======================================================
# 第一部分：智能优化算法 (日前经济调度)
# =======================================================
def run_economic_optimization():
    print(">>> 步骤 1/4: 正在运行 Pulp 经济优化算法...")
    hours = list(range(24))
    dt = 1.0

    # 典型日总需冷量 (kW冷)
    Q_load = {
        0: 6759.02, 1: 6833.49, 2: 6756.07, 3: 6761.13, 4: 6735.04, 5: 6782.21,
        6: 6736.83, 7: 6907.68, 8: 7278.19, 9: 8062.28, 10: 8739.60, 11: 8618.55,
        12: 8602.05, 13: 8499.88, 14: 8447.85, 15: 7561.75, 16: 7335.91, 17: 7369.50,
        18: 7425.73, 19: 7146.04, 20: 7373.37, 21: 7078.45, 22: 7373.73, 23: 7136.02
    }

    # 分时电价
    price = {}
    for h in hours:
        if 0 <= h < 8: price[h] = 0.2203
        elif h in [9, 10, 15, 16]: price[h] = 1.1477
        elif h in [8, 11, 12, 17, 18, 19, 20, 21, 22]: price[h] = 0.9564
        else: price[h] = 0.5796

    chillers = {
        "C1": {"Qmax": 7032, "cop_25": 4.388, "cop_50": 5.167, "cop_75": 5.583, "cop_100": 5.599},
        "C2": {"Qmax": 3516, "cop_25": 4.695, "cop_50": 5.543, "cop_75": 5.760, "cop_100": 5.390},
        "C3": {"Qmax": 3517, "cop_25": 6.620, "cop_50": 7.520, "cop_75": 7.430, "cop_100": 6.260},
        "C4": {"Qmax": 2210, "cop_25": 3.330, "cop_50": 4.720, "cop_75": 5.580, "cop_100": 6.020},
    }
    for c, d in chillers.items():
        d["seg_cop"] = [d["cop_25"], d["cop_50"], d["cop_75"], d["cop_100"]]

    Q_ice_charge_max = 5016.0; P_ice_charge_max = 1182.0; COP_ice_charge = 4.24
    E_ice_max = 21000 * 3.517; Q_discharge_max = 0.13 * E_ice_max
    
    segments = [("s1", 0.25), ("s2", 0.25), ("s3", 0.25), ("s4", 0.25)]

    # 建立模型
    m = pulp.LpProblem("IceStorage_Optimization", pulp.LpMinimize)

    # 决策变量
    q_ch = {(c, h): pulp.LpVariable(f"q_{c}_{h}", lowBound=0) for c in chillers for h in hours}
    p_ch = {(c, h): pulp.LpVariable(f"p_{c}_{h}", lowBound=0) for c in chillers for h in hours}
    q_seg = {(c, h, s): pulp.LpVariable(f"qseg_{c}_{h}_{s}", lowBound=0) for c in chillers for h in hours for s,_ in segments}
    q_ice_ch = {h: pulp.LpVariable(f"q_ice_ch_{h}", lowBound=0, upBound=Q_ice_charge_max) for h in hours}
    p_ice_ch = {h: pulp.LpVariable(f"p_ice_ch_{h}", lowBound=0, upBound=P_ice_charge_max) for h in hours}
    q_dis = {h: pulp.LpVariable(f"q_dis_{h}", lowBound=0, upBound=Q_discharge_max) for h in hours}
    soc = {h: pulp.LpVariable(f"soc_{h}", lowBound=0, upBound=E_ice_max) for h in hours}
    p_total = {h: pulp.LpVariable(f"p_total_{h}", lowBound=0) for h in hours}

    # 约束条件
    for c, d in chillers.items():
        for h in hours:
            m += q_ch[(c,h)] == pulp.lpSum(q_seg[(c,h,s)] for s,_ in segments)
            for idx, (s, rat) in enumerate(segments):
                m += q_seg[(c,h,s)] <= d["Qmax"] * rat
            m += p_ch[(c,h)] == pulp.lpSum(q_seg[(c,h,s)] / d["seg_cop"][idx] for idx, (s,_) in enumerate(segments))

    for h in hours:
        m += q_ice_ch[h] == COP_ice_charge * p_ice_ch[h]
        m += pulp.lpSum(q_ch[(c,h)] for c in chillers) + q_dis[h] == Q_load[h] + q_ice_ch[h]
        if h == 0: m += soc[h] == soc[23] + q_ice_ch[h]*dt - q_dis[h]*dt
        else:      m += soc[h] == soc[h-1] + q_ice_ch[h]*dt - q_dis[h]*dt
        m += p_total[h] == pulp.lpSum(p_ch[(c,h)] for c in chillers) + p_ice_ch[h]

    # 目标函数 (最小化日电费)
    m += pulp.lpSum(price[h] * p_total[h] * dt for h in hours)
    
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    print(f"优化完成！状态: {pulp.LpStatus[m.status]}")

    # 提取结果到 numpy 数组
    total_q_load = np.array([Q_load[h] for h in hours])
    total_q_chiller = np.array([sum(q_ch[(c,h)].value() for c in chillers) for h in hours])
    q_ice_charge = np.array([q_ice_ch[h].value() for h in hours])
    q_discharge = np.array([q_dis[h].value() for h in hours])
    
    return total_q_load, total_q_chiller, q_ice_charge, q_discharge

# =======================================================
# 第二部分：Simulink 物理管网动态仿真
# =======================================================
def main():
    # 1. 获取优化结果 (长度为 24 的数组，单位 kW)
    q_load_24, q_ch_total_24, q_ice_ch_24, q_dis_24 = run_economic_optimization()
    
    print(">>> 步骤 2/4: 正在处理数据适配 Simulink...")
    # 将长度 24 的数据延长为 25 (末端补齐 t=24 的点)
    def extend_to_25(arr): return np.append(arr, arr[-1])
    
    q_load_25 = extend_to_25(q_load_24)
    q_ch_total_25 = extend_to_25(q_ch_total_24)
    q_ice_ch_25 = extend_to_25(q_ice_ch_24)
    q_dis_25 = extend_to_25(q_dis_24)

    # ---------------- 物理输入构造 ----------------
    # A. 供冷源转换 (kW 转 W，且必须为负数代表制冷)
    # 真正输入管网的制冷量 = 机组总制冷 - 用于制冰的冷量
    sim_chiller_W = -1.0 * (q_ch_total_25 - q_ice_ch_25) * 1000.0
    sim_tank_W = -1.0 * q_dis_25 * 1000.0

    # B. 用户热负荷分配 (将总负荷拆分给 6 个用户，正数代表吸热)
    # 假设各用户规模比例: 10%, 10%, 20%, 30%, 10%, 20%
    ratios = [0.1, 0.1, 0.2, 0.3, 0.1, 0.2]
    q_load_W = q_load_25 * 1000.0
    
    heat_users = [q_load_W * r for r in ratios]
    
    # C. 用户动态流量分配 (根据 Q = c*m*dT, 设 dT=5度)
    flow_users = [h / 21000.0 for h in heat_users]

    # D. 软启动处理 (极其重要，防止水力冲击报错)
    sim_chiller_W[0] = 0.0
    sim_tank_W[0] = 0.0
    for i in range(6):
        heat_users[i][0] = 0.0
        flow_users[i][0] = 0.0

    # ---------------- 准备推送 ----------------
    print(">>> 步骤 3/4: 启动 MATLAB Engine，推送 24 小时动态数据...")
    eng = matlab.engine.start_matlab()
    t = np.linspace(0, 24*3600, 25)
    
    def fmt(val_array): return matlab.double(np.column_stack((t, val_array)).tolist())

    eng.workspace['chiller'] = fmt(sim_chiller_W)
    eng.workspace['tank']    = fmt(sim_tank_W)
    for i in range(6):
        eng.workspace[f'user{i+1}_heat'] = fmt(heat_users[i])
        eng.workspace[f'user{i+1}_flow'] = fmt(flow_users[i])

    # ---------------- 运行仿真 ----------------
    model_name = 'guanwangmoxing_2025a'
    print(f">>> 步骤 4/4: 正在运行 Simulink 物理仿真: {model_name}...")
    try:
        eng.eval(f"load_system('{model_name}')", nargout=0)
        eng.set_param(model_name, 'StopTime', str(24*3600), nargout=0)
        eng.eval(f"out = sim('{model_name}');", nargout=0)
        print("仿真完美结束！")

        # ---------------- 提取数据绘图 ----------------
        tout = np.array(eng.eval("out.tout")).flatten()
        pump_power = np.array(eng.eval("out.PumpPower")).flatten()
        
        # 绘制优化结果和泵能耗对比图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # 上图：冷量供需平衡可视化
        ax1.plot(t/3600, q_load_25, 'k--', label='Total Cooling Demand')
        ax1.bar(t/3600, -sim_chiller_W/1000, width=0.8, color='blue', alpha=0.6, label='Chiller Supply (to Network)')
        ax1.bar(t/3600, -sim_tank_W/1000, width=0.8, bottom=-sim_chiller_W/1000, color='cyan', alpha=0.6, label='Ice Tank Discharge')
        ax1.set_ylabel('Cooling Power (kW)')
        ax1.set_title('AI Optimized Cooling Dispatch Strategy (24H)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 下图：Simulink 物理引擎反馈的泵能耗
        ax2.plot(tout/3600, pump_power/1000, color='red', linewidth=2, label='Hydraulic Pump Power')
        ax2.set_xlabel('Time (Hours)')
        ax2.set_ylabel('Pump Power (kW)')
        ax2.set_title('Simscape Simulated Pump Energy Consumption')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 24])

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print("仿真运行失败:", e)
        
    eng.quit()

if __name__ == '__main__':
    main()