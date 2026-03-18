import numpy as np


class TreePumpSystemSimulator:
    def __init__(self, pump_params, branch_params, tree_structure):
        """
        初始化树状泵系统模拟器
        :param pump_params: 泵机参数字典，包含：
            - rated_flow: 额定流量 (m^3/h)
            - rated_head: 额定扬程 (m)
            - rated_power: 额定功率 (kW)
            - speed: 当前泵的转速 (rpm)
            - efficiency_curve: 功率效率曲线函数，输入为相对流量 (Q/Q_rated)，返回当前效率
        :param branch_params: 每条支路参数，列表，每个元素为支路字典，包含：
            - "name": 唯一的支路名称
            - "length": 管道长度 (m)
            - "diameter": 管径 (m)
            - "user_cooling_demand": 用户制冷量需求 (kW)
            - "pipe_insulation": 是否有保温 (True/False)
        :param tree_structure: 树状网络结构，字典定义支路连接关系，key为父节点名，value为子节点列表
        """
        self.pump_params = pump_params
        self.branch_params = {branch["name"]: branch for branch in branch_params}  # 转化为方便查询的字典
        self.tree_structure = tree_structure
        self.update_hydraulic_properties()

    def update_hydraulic_properties(self):
        """根据支路参数，计算水头损失系数及热损失参数"""
        for branch in self.branch_params.values():
            # 水力学参数：摩擦系数和沿程阻力  h_f ∝ f * (L/D) * v²
            length = branch["length"]
            diameter = branch["diameter"]
            f = 0.02  # 假设管道摩擦系数为0.02
            branch["friction_constant"] = f * length / diameter

            # 热力学参数：假设沿管道每米的热损失
            if branch["pipe_insulation"]:  # 如果管道有保温层
                heat_loss_rate = 5  # W/m（较低的热损失）
            else:  # 无保温情况下热损失上升
                heat_loss_rate = 20  # W/m

            branch["heat_loss_rate"] = heat_loss_rate / 1000  # 转化为 kW/m

    def calculate_heat_flow_distribution(self, supply_temp, return_temp, node):
        """
        递归计算树状网络中每条支路的流量、热损及回水温度
        :param supply_temp: 根节点供水温度
        :param return_temp: 末端回水温度
        :param node: 当前计算的支路
        :return: 当前支路的总流量和回水温度
        """
        downstream_branches = self.tree_structure.get(node, [])  # 获取当前支路的下游分支

        if not downstream_branches:  # 如果是叶节点（即用户）
            branch = self.branch_params[node]
            delta_t = return_temp - supply_temp  # 定义供回水温差
            heat_loss_total = branch["heat_loss_rate"] * branch["length"]  # 当前支路热损失总量 kW
            adjusted_cooling_demand = branch["user_cooling_demand"] + heat_loss_total  # 考虑热损失的冷量总需求

            # 流量 = 用冷量 / (c_p * ΔT)
            branch["flow_rate"] = adjusted_cooling_demand / (4.2 * delta_t)  # m^3/s

            # 返回调整后的供回水温度
            branch["supply_temp_loss"] = supply_temp
            branch["return_temp_loss"] = return_temp - (heat_loss_total / (4.2 * branch["flow_rate"])) if branch["flow_rate"] > 0 else return_temp
            return branch["flow_rate"], branch["return_temp_loss"]  # 返回流量与回水温度

        # 对非叶节点，递归计算其下游分支的流量和热平衡
        total_flow_rate = 0
        weighted_downstream_temp = 0

        for downstream in downstream_branches:
            flow, downstream_temp = self.calculate_heat_flow_distribution(supply_temp, return_temp, downstream)
            total_flow_rate += flow
            weighted_downstream_temp += flow * downstream_temp

        # 当前支路的回水温度 = 加权平均回水温度
        branch = self.branch_params[node]
        branch["flow_rate"] = total_flow_rate  # 本支路的流量为下游总流量
        branch["return_temp_loss"] = weighted_downstream_temp / total_flow_rate if total_flow_rate > 0 else return_temp

        heat_loss_total = branch["heat_loss_rate"] * branch["length"]
        branch["supply_temp_loss"] = supply_temp
        branch["return_temp_loss"] -= heat_loss_total / (4.2 * branch["flow_rate"]) if branch["flow_rate"] > 0 else return_temp

        return branch["flow_rate"], branch["return_temp_loss"]

    def calculate_pump_performance(self):
        """计算水泵性能，包括扬程、功率和效率"""
        total_flow_rate = self.branch_params["root"]["flow_rate"]  # 获取根节点泵的进水总流量 (m^3/s)
        total_flow_rate_m3h = total_flow_rate * 3600  # 转化为 m^3/h

        speed_ratio = self.pump_params["speed"] / 2900  # 以额定转速（2900rpm）为基准
        max_head = self.pump_params["rated_head"] * (speed_ratio ** 2)
        max_flow = self.pump_params["rated_flow"] * speed_ratio

        # 限制流量超出额定范围
        if total_flow_rate_m3h > max_flow:
            total_flow_rate_m3h = max_flow

        # 耗电功率和效率
        power_curve = (total_flow_rate_m3h / self.pump_params["rated_flow"]) ** 3 * self.pump_params["rated_power"]
        efficiency_curve = max(0, -0.3 * (total_flow_rate_m3h / self.pump_params["rated_flow"] - 1) ** 2 + 0.8)
        actual_power = power_curve / efficiency_curve if efficiency_curve > 0 else 0

        self.performance = {
            "total_flow_rate": total_flow_rate_m3h,
            "total_power": actual_power,
            "head": max_head,
            "efficiency": efficiency_curve,
        }

    def simulate(self, supply_temp, return_temp):
        """运行树状系统模拟"""
        self.calculate_heat_flow_distribution(supply_temp, return_temp, node="root")
        self.calculate_pump_performance()

        print("\n=== Simulation Results ===")
        for branch_name, branch in self.branch_params.items():
            print(f"{branch_name}:")
            print(f"  Flow Rate = {branch['flow_rate'] * 3600:.2f} m^3/h")
            print(f"  Supply Temp (after loss) = {branch['supply_temp_loss']:.2f} °C")
            print(f"  Return Temp (after loss) = {branch['return_temp_loss']:.2f} °C")
        print(f"Total Pump Flow Rate: {self.performance['total_flow_rate']:.2f} m^3/h")
        print(f"Pump Power Consumption: {self.performance['total_power']:.2f} kW")
        print(f"Pump Head: {self.performance['head']:.2f} m")
        print(f"Pump Efficiency: {self.performance['efficiency'] * 100:.2f}%\n")


# 示例：树状泵系统参数初始化
pump_params = {
    "rated_flow": 100,     # 额定流量 (m^3/h)
    "rated_head": 30,      # 额定扬程 (m)
    "rated_power": 22,     # 额定功率 (kW)
    "speed": 2900,         # 当前转速 (rpm)
    "efficiency_curve": lambda x: -0.3 * (x - 1)**2 + 0.8  # 假设泵效率曲线
}

# 支路参数
branch_params = [
    {"name": "root", "length": 0, "diameter": 0.6, "user_cooling_demand": 0, "pipe_insulation": True},
    {"name": "DN600_top", "length": 400, "diameter": 0.6, "user_cooling_demand": 0, "pipe_insulation": True},
    {"name": "DN600_bottom", "length": 300, "diameter": 0.6, "user_cooling_demand": 0, "pipe_insulation": False},
    {"name": "省药检局", "length": 120, "diameter": 0.2, "user_cooling_demand": 50, "pipe_insulation": True},
    {"name": "天境生物", "length": 150, "diameter": 0.15, "user_cooling_demand": 30, "pipe_insulation": True},
    {"name": "接3站", "length": 200, "diameter": 0.15, "user_cooling_demand": 20, "pipe_insulation": False},
    {"name": "和泽", "length": 280, "diameter": 0.25, "user_cooling_demand": 60, "pipe_insulation": True},
    {"name": "加速五期", "length": 300, "diameter": 0.2, "user_cooling_demand": 40, "pipe_insulation": False},
]

# 树状网络
tree_structure = {
    "root": ["DN600_top", "DN600_bottom"],
    "DN600_top": ["省药检局", "天境生物", "接3站"],
    "DN600_bottom": ["和泽", "加速五期"],
}

# 创建模拟器实例
simulator = TreePumpSystemSimulator(pump_params, branch_params, tree_structure)

# 流体温度设置
supply_temp = 7    # 供水温度 (°C)
return_temp = 12   # 回水温度 (°C)

# 开始模拟
simulator.simulate(supply_temp, return_temp)

# 增加支路用户需求和动态调整泵速
simulator.set_user_cooling_demand("天境生物", 50)  # 增加天境生物的用冷需求
simulator.set_pump_speed(2500)  # ��低泵的转速以降低能耗
simulator.simulate(supply_temp, return_temp)