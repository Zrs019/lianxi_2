import numpy as np
import scipy.io as sio

# 定义管网结构
pipelines = {
    1: {"length": 500, "diameter": 0.3, "flow": 0, "thermal_loss": 0},  # 管道1
    2: {"length": 300, "diameter": 0.25, "flow": 0, "thermal_loss": 0},  # 管道2
    3: {"length": 700, "diameter": 0.2, "flow": 0, "thermal_loss": 0},  # 管道3
}

users = {
    "UserA": {"demand": 5},  # 用户A的冷量需求 (m³/h)
    "UserB": {"demand": 10}, # 用户B的冷量需求
}

# 固定参数
pump_efficiency = 0.8  # 泵效率（80%）
g = 9.81  # 重力加速度
density = 1000  # 水密度 (kg/m³)
c_p = 4.18  # 水的比热容 (kJ/kg°C)

# 计算每段管道的水头损失及热量损失 (q_loss)
def calculate_pipeline_parameters():
    for key, pipe in pipelines.items():
        flow = sum([u["demand"] for u in users.values()])  # 流量 (假设所有用户共享管道)
        v = flow / (np.pi * (pipe["diameter"] / 2) ** 2)  # 流速
        f = 0.02  # Darcy摩擦系数
        pipe["head_loss"] = f * (pipe["length"] / pipe["diameter"]) * (v ** 2) / (2 * g)
        q_loss = 0.5 * np.pi * pipe["diameter"] * pipe["length"] * 5  # 热损失，假设传热系数为0.5
        pipe["thermal_loss"] = q_loss

# 保存计算结果
def save_results_to_matlab():
    sio.savemat("pipeline_simulation.mat", {
        "pipes": np.array([[p["length"], p["diameter"], p["head_loss"], p["thermal_loss"]] for p in pipelines.values()]),
        "pump_efficiency": pump_efficiency,
        "user_demand": np.array([u["demand"] for u in users.values()])
    })

# 主程序
calculate_pipeline_parameters()
save_results_to_matlab()
print("Simulation data saved to 'pipeline_simulation.mat'")