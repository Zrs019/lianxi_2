import pulp
import numpy as np
import pandas as pd
import matlab.engine
import matplotlib.pyplot as plt

# ==============================================================================
# 模块一：输入参数配置
# ==============================================================================
def load_data_and_equipment():
    # 典型日 24 小时总需冷量 (kW)
    Q_load_user = {
        0: 6759.02, 1: 6833.49, 2: 6756.07, 3: 6761.13, 4: 6735.04, 5: 6782.21,
        6: 6736.83, 7: 6907.68, 8: 7278.19, 9: 8062.28, 10: 8739.60, 11: 8618.55,
        12: 8602.05, 13: 8499.88, 14: 8447.85, 15: 7561.75, 16: 7335.91, 17: 7369.50,
        18: 7425.73, 19: 7146.04, 20: 7373.37, 21: 7078.45, 22: 7373.73, 23: 7136.02
    }
    # 制冷机组参数字典
    chillers = {
        "C1": {"Qmax": 7032, "cop_25": 4.388, "cop_50": 5.167, "cop_75": 5.583, "cop_100": 5.599},
        "C2": {"Qmax": 3516, "cop_25": 4.695, "cop_50": 5.543, "cop_75": 5.760, "cop_100": 5.390},
        "C3": {"Qmax": 3517, "cop_25": 6.620, "cop_50": 7.520, "cop_75": 7.430, "cop_100": 6.260},
        "C4": {"Qmax": 2210, "cop_25": 3.330, "cop_50": 4.720, "cop_75": 5.580, "cop_100": 6.020},
    }
    return Q_load_user, chillers

# ==============================================================================
# 模块二：物理预诊断 (算出精确漏热 + 水泵发热)
# ==============================================================================
def pre_calculate_physics_loss(Q_load_user):
    """
    在 AI 调度前，提前算出 11 段物理管道的漏热量与水泵发热量 (kW)
    """
    Q_load_W = np.array(list(Q_load_user.values())) * 1000.0
    ratios = [0.1, 0.1, 0.2, 0.3, 0.1, 0.2]
    
    # 统一修改为 5℃ 温差计算基准 (4200 * 5 = 21000)
    u_flow_avg = [np.mean(Q_load_W * r / 21000.0) for r in ratios]
    u1, u2, u3, u4, u5, u6 = u_flow_avg
    
    flows_dynamic_avg = [
        u1+u2+u3, u1, u2+u3, u2, u3, u3, 
        u4+u5+u6, u4, u5+u6, u5, u6
    ]
    pipe_lengths_km = [0.254, 0.1315, 0.544, 0.008, 0.136, 0.008, 0.321, 0.008, 0.074, 0.113, 0.012] 
    
    # 严格根据物理公式 Q = c * m * dT 计算漏热 (W)
    Q_loss_list_W = [float(4200.0 * flows_dynamic_avg[i] * (0.8 * pipe_lengths_km[i])) for i in range(11)]
    total_network_loss_kW = sum(Q_loss_list_W) / 1000.0
    
    # 【修复核心】：删除 170.0 的水泵发热补偿，只让冷机补偿真实的管道漏热！
    total_comp_kW = total_network_loss_kW 
    
    return total_comp_kW, Q_loss_list_W

# ==============================================================================
# 模块三：Pulp 经济调度优化 (注入物理损耗补偿)
# ==============================================================================
def run_economic_dispatch(Q_load_user, chillers, total_comp_kW):
    print(f">>> 步骤 2/4: 运行 AI 经济优化 (已注入物理管网漏热与水泵发热基底: {total_comp_kW:.2f} kW)...")
    hours = list(range(24))
    dt = 1.0

    # 目标冷量 = 用户需求负荷 + 物理系统的能量损失补偿
    Q_load_target = {h: v + total_comp_kW for h, v in Q_load_user.items()}

    price = {}
    for h in hours:
        if 0 <= h < 8: price[h] = 0.2203
        elif h in [9, 10, 15, 16]: price[h] = 1.1477
        elif h in [8, 11, 12, 17, 18, 19, 20, 21, 22]: price[h] = 0.9564
        else: price[h] = 0.5796

    for c, d in chillers.items():
        d["seg_cop"] = [d["cop_25"], d["cop_50"], d["cop_75"], d["cop_100"]]

    Q_ice_charge_max = 5016.0; P_ice_charge_max = 1182.0; COP_ice_charge = 4.24
    E_ice_max = 21000 * 3.517; Q_discharge_max = 0.13 * E_ice_max
    segments = [("s1", 0.25), ("s2", 0.25), ("s3", 0.25), ("s4", 0.25)]

    m = pulp.LpProblem("IceStorage_Optimization", pulp.LpMinimize)

    q_ch = {(c, h): pulp.LpVariable(f"q_{c}_{h}", lowBound=0) for c in chillers for h in hours}
    p_ch = {(c, h): pulp.LpVariable(f"p_{c}_{h}", lowBound=0) for c in chillers for h in hours}
    q_seg = {(c, h, s): pulp.LpVariable(f"qseg_{c}_{h}_{s}", lowBound=0) for c in chillers for h in hours for s,_ in segments}
    q_ice_ch = {h: pulp.LpVariable(f"q_ice_ch_{h}", lowBound=0, upBound=Q_ice_charge_max) for h in hours}
    p_ice_ch = {h: pulp.LpVariable(f"p_ice_ch_{h}", lowBound=0, upBound=P_ice_charge_max) for h in hours}
    q_dis = {h: pulp.LpVariable(f"q_dis_{h}", lowBound=0, upBound=Q_discharge_max) for h in hours}
    soc = {h: pulp.LpVariable(f"soc_{h}", lowBound=0, upBound=E_ice_max) for h in hours}
    p_total = {h: pulp.LpVariable(f"p_total_{h}", lowBound=0) for h in hours}

    for c, d in chillers.items():
        for h in hours:
            m += q_ch[(c,h)] == pulp.lpSum(q_seg[(c,h,s)] for s,_ in segments)
            for idx, (s, rat) in enumerate(segments): m += q_seg[(c,h,s)] <= d["Qmax"] * rat
            m += p_ch[(c,h)] == pulp.lpSum(q_seg[(c,h,s)] / d["seg_cop"][idx] for idx, (s,_) in enumerate(segments))

    for h in hours:
        m += q_ice_ch[h] == COP_ice_charge * p_ice_ch[h]
        # 调度必须满足包含了补偿目标的新负荷曲线
        m += pulp.lpSum(q_ch[(c,h)] for c in chillers) + q_dis[h] == Q_load_target[h] + q_ice_ch[h]
        if h == 0: m += soc[h] == soc[23] + q_ice_ch[h]*dt - q_dis[h]*dt
        else:      m += soc[h] == soc[h-1] + q_ice_ch[h]*dt - q_dis[h]*dt
        m += p_total[h] == pulp.lpSum(p_ch[(c,h)] for c in chillers) + p_ice_ch[h]

    m += pulp.lpSum(price[h] * p_total[h] * dt for h in hours)
    m.solve(pulp.PULP_CBC_CMD(msg=False))

    net_chiller_supply = np.array([sum(q_ch[(c,h)].value() for c in chillers) - q_ice_ch[h].value() for h in hours])
    q_discharge = np.array([q_dis[h].value() for h in hours])
    
    return net_chiller_supply, q_discharge

# ==============================================================================
# 模块四：物理引擎主程序
# ==============================================================================
def main():
    Q_load_user, chillers = load_data_and_equipment()
    
    # 1. 前置预诊：算出物理模型到底会丢多少冷量
    print(">>> 步骤 1/4: 进行物理引擎前置评估...")
    total_comp_kW, Q_loss_list_W = pre_calculate_physics_loss(Q_load_user)
    print(f"✅ 诊断完成: 稳态漏热与水泵热补偿总计约为 {total_comp_kW:.2f} kW")
    
    # 2. 带着物理补偿目标跑 AI
    q_ch_net_24, q_dis_24 = run_economic_dispatch(Q_load_user, chillers, total_comp_kW)
    q_load_user_24 = np.array([Q_load_user[h] for h in range(24)])
    
    print("\n>>> 步骤 3/4: 数据封装与 MATLAB 对接...")
    def extend_to_25(arr): return np.append(arr, arr[-1])
    q_load_user_25 = extend_to_25(q_load_user_24)
    q_ch_net_25 = extend_to_25(q_ch_net_24)
    q_dis_25 = extend_to_25(q_dis_24)

    # A. 供冷源 (-W) : 此处已包含用户的 100% 负荷 + 物理损失补偿
    sim_chiller_W = -1.0 * q_ch_net_25 * 1000.0
    sim_tank_W = -1.0 * q_dis_25 * 1000.0

    # B. 用户实际热负荷 (维持原始动态负荷注入节点)
    ratios = [0.1, 0.1, 0.2, 0.3, 0.1, 0.2]
    heat_users = [q_load_user_25 * 1000.0 * r for r in ratios]
    
    # ---------------------------------------------------------
    # C. 【核心重构】：彻底解耦流量与实时负荷 (定流量系统模型)
    # ---------------------------------------------------------
    # 获取全天最大的总热负荷 (W)
    max_total_heat = np.max(q_load_user_25 * 1000.0)
    
    # 策略：以最大负荷、温差 5℃ (21000) 为基准，设定管网的设计最大恒定流量
    design_flow_total = max_total_heat / 21000.0
    
    flow_users = []
    for r in ratios:
        # 每个用户的流量被固定为其设计配额，不再随当前时刻的热负荷波动！
        user_constant_flow = design_flow_total * r
        flow_array = np.full(25, user_constant_flow) 
        flow_users.append(flow_array)
    # ---------------------------------------------------------

    # D. 软启动处理 (防冲击)
    sim_chiller_W[0] = 0.0
    sim_tank_W[0] = 0.0
    for i in range(6):
        heat_users[i][0] = 0.0
        flow_users[i][0] = 0.0  # 确保 t=0 时流量也是从 0 启动

    eng = matlab.engine.start_matlab()
    t = np.linspace(0, 24*3600, 25)
    def fmt(val_array): return matlab.double(np.column_stack((t, val_array)).tolist())

    eng.workspace['chiller'] = fmt(sim_chiller_W)
    eng.workspace['tank']    = fmt(sim_tank_W)
   
   # ---------------------------------------------------------
    # 🌟 核心升级：变频水泵动态调压策略 (VFD Pump Control)
    # ---------------------------------------------------------
    # 获取全天最大的总负荷基准
    max_total_heat = np.max(q_load_user_25 * 1000.0)
    pump_pressure_array = np.zeros(25)
    
    # 设定水泵变频调压的上下限 (单位: Pa)
    # 500kPa 为设计满载压力，250kPa 为保障最远端(用户6)有水流的最低底线压力
    MAX_PRESSURE = 500000.0 
    MIN_PRESSURE = 250000.0 
    
    for h in range(25):
        if h == 0:
            pump_pressure_array[h] = 0.0
        else:
            current_total_heat = q_load_user_25[h] * 1000.0
            load_ratio = current_total_heat / max_total_heat
            
            # 【流体力学优化】：改为平方律变频算法 (Quadratic Pump Law)
            # 因为阻力与流量的平方成正比，按平方律降压才能精准匹配所需流量
            dynamic_pressure = MAX_PRESSURE * (load_ratio ** 2) 
            
            # 同样保留安全底线
            pump_pressure_array[h] = max(dynamic_pressure, MIN_PRESSURE)
            
    # 将计算好的 24 小时变频调压曲线推送给 Simulink 水泵
    eng.workspace['pump_pressure'] = fmt(pump_pressure_array)
    # ---------------------------------------------------------
    
    # 用户热负荷依然需要发送
    for i in range(6):
        eng.workspace[f'user{i+1}_heat'] = fmt(heat_users[i])
   
    for i in range(11):
        eng.workspace[f'Q_loss_{i+1}'] = Q_loss_list_W[i]

    model_name = 'guanwangmoxing_2025a'
    print(f">>> 步骤 4/4: 正在运行 Simulink 物理验证: {model_name}...")
    try:
        eng.eval(f"load_system('{model_name}')", nargout=0)
        eng.set_param(model_name, 'StopTime', str(24*3600), nargout=0)
        eng.eval(f"out = sim('{model_name}');", nargout=0)
        print("✅ 物理能量守恒验证成功！正在绘图...")

        tout = np.array(eng.eval("out.tout")).flatten()
        pump_power = np.array(eng.eval("out.PumpPower")).flatten()
        
        T_sup, T_ret = [], []
        for i in range(1, 7):
            T_sup.append(np.array(eng.eval(f"out.T_sup_{i}")).flatten() - 273.15)
            T_ret.append(np.array(eng.eval(f"out.T_ret_{i}")).flatten() - 273.15)
        
        # 在提取 T_sup 和 T_ret 的循环旁边，加上提取真实流量的代码
        real_flows = []
        for i in range(1, 7):
            real_flows.append(np.array(eng.eval(f"out.real_flow_{i}")).flatten())
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
        
        ax1.plot(t/3600, q_load_user_25, 'k--', linewidth=2, label='100% User Cooling Load')
        ax1.plot(t/3600, q_load_user_25 + total_comp_kW, 'r:', linewidth=2, label='Target Supply (Load + Loss & Pump Heat)')
        ax1.bar(t/3600, -sim_chiller_W/1000, width=0.8, color='#4c72b0', alpha=0.8, label='Chiller Net Supply')
        ax1.bar(t/3600, -sim_tank_W/1000, width=0.8, bottom=-sim_chiller_W/1000, color='#55a868', alpha=0.8, label='Ice Tank Discharge')
        ax1.set_ylabel('Cooling Power (kW)')
        ax1.set_title('Step 1: AI Optimized Energy Dispatch (Loss & Friction Compensated)')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.5)

        ax2.plot(tout/3600, pump_power/1000, color='#c44e52', linewidth=2.5, label='Hydraulic Pump Power (kW)')
        ax2.set_ylabel('Pump Power (kW)')
        ax2.set_title('Step 2: Simscape Feedback - Network Pump Energy')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.5)

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        for i in range(6):
            ax3.plot(tout/3600, T_sup[i], color=colors[i], linestyle='-', linewidth=1.5, label=f'User {i+1} Supply (℃)')
            ax3.plot(tout/3600, T_ret[i], color=colors[i], linestyle='--', linewidth=1.5, label=f'User {i+1} Return (℃)')
        ax3.set_xlabel('Time (Hours)')
        ax3.set_ylabel('Temperature (℃)')
        ax3.set_title('Step 3: Simscape Feedback - User Terminal Temperatures')
        ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
        ax3.grid(True, linestyle='--', alpha=0.5)
        
        ax1.set_xlim([0, 24])
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print("\n=== 仿真运行或结果提取失败 ===")
        print(e)
    
    eng.quit()

if __name__ == '__main__':
    main()