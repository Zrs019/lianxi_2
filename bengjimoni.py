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
        self.update_hydraulic_properties()  # 初始化水力和热力学属性

    def update_hydraulic_properties(self):
        """初始化水力学特性和热损失系数"""
        for branch in self.branch_params.values():
            # 计算水力阻力
            length = branch["length"]
            diameter = branch["diameter"]
            f = 0.02  # 假设平均摩擦系数
            branch["friction_constant"] = f * length / diameter  # 摩擦阻力系数
            
            # 热损失计算
            if branch.get("pipe_insulation", True):  # 有保温管
                heat_loss_rate = 5  # W/m (保温)
            else:
                heat_loss_rate = 20  # W/m (无保温)
            
            branch["heat_loss_rate"] = heat_loss_rate / 1000  # 转换成 kW/m

    def calculate_heat_and_flow(self, node, supply_temp, return_temp):
        """
        递归计算从树的叶节点到根节点的流量和热损失
        :param node: 当前支路节点名
        :param supply_temp: 供水温度 (°C)
        :param return_temp: 回水温度 (°C)
        :return: 当前支路的总流量和平均回水温度
        """
        downstream_branches = self.tree_structure.get(node, [])  # 获取下游分支

        if not downstream_branches:  # 如果是叶节点（用户节点）
            branch = self.branch_params[node]
            delta_t = return_temp - supply_temp  # 当前供回水温差
            heat_loss_total = branch["heat_loss_rate"] * branch["length"]  # 总热损失 (kW)
            adjusted_cooling_demand = branch["user_cooling_demand"] + heat_loss_total  # 总冷量需求

            # 计算流量 = 用冷量 / (比热 * 温差)
            branch["flow_rate"] = adjusted_cooling_demand / (4.2 * delta_t)  # m³/s

            # 计算温差的变化
            branch["supply_temp_loss"] = supply_temp + (
                heat_loss_total / (4.2 * branch["flow_rate"]) if branch["flow_rate"] > 0 else 0
            )
            branch["return_temp_loss"] = return_temp - (
                heat_loss_total / (4.2 * branch["flow_rate"]) if branch["flow_rate"] > 0 else 0
            )

            return branch["flow_rate"], branch["return_temp_loss"]

        # 如果是中间节点，递归处理下游节点
        total_flow_rate = 0  # 下游总流量
        weighted_temp = 0  # 加权回水温度求和

        for downstream in downstream_branches:
            flow, downstream_temp = self.calculate_heat_and_flow(
                downstream, supply_temp, return_temp
            )
            total_flow_rate += flow
            weighted_temp += flow * downstream_temp

        # 当前支路的回水温度 = 加权平均回水温度
        branch = self.branch_params[node]
        branch["flow_rate"] = total_flow_rate  # 当前支路流量为所有下游流量总和
        branch["return_temp_loss"] = weighted_temp / total_flow_rate if total_flow_rate > 0 else return_temp

        # 计算热损失对供水/回水温度的影响
        heat_loss_total = branch["heat_loss_rate"] * branch["length"]
        branch["supply_temp_loss"] = supply_temp
        branch["return_temp_loss"] -= heat_loss_total / (4.2 * branch["flow_rate"]) if branch["flow_rate"] > 0 else 0

        return branch["flow_rate"], branch["return_temp_loss"]

    def calculate_pump_performance(self):
        """根据总流量计算泵的性能"""
        total_flow_rate = self.branch_params["root"]["flow_rate"]  # 根节点的流量为总流量
        total_flow_rate_m3h = total_flow_rate * 3600  # 转换为 m³/h

        speed_ratio = self.pump_params["speed"] / 2900  # 转速调整
        max_head = self.pump_params["rated_head"] * (speed_ratio ** 2)  # 扬程变化
        max_flow = self.pump_params["rated_flow"] * speed_ratio  # 最大流量限制

        # 限制超出额定限制的流量
        if total_flow_rate_m3h > max_flow:
            total_flow_rate_m3h = max_flow

        # 立方关系功率曲线和效率
        power_curve = (total_flow_rate_m3h / self.pump_params["rated_flow"]) ** 3 * self.pump_params["rated_power"]
        efficiency_curve = max(0, -0.3 * (total_flow_rate_m3h / self.pump_params["rated_flow"] - 1) ** 2 + 0.8)
        actual_power = power_curve / efficiency_curve if efficiency_curve > 0 else 0  # 实际功率

        self.performance = {
            "total_flow_rate": total_flow_rate_m3h,
            "total_power": actual_power,
            "head": max_head,
            "efficiency": efficiency_curve,
        }

    def simulate(self, supply_temp, return_temp):
        """运行模拟"""
        self.calculate_heat_and_flow("root", supply_temp, return_temp)
        self.calculate_pump_performance()

        print("\n=== 模拟结果 ===")
        for branch_name, branch in self.branch_params.items():
            print(f"{branch_name}:")
            print(f"  流量 = {branch['flow_rate'] * 3600:.2f} m³/h")
            print(f"  供水温度(含热损) = {branch['supply_temp_loss']:.2f} °C")
            print(f"  回水温度(含热损) = {branch['return_temp_loss']:.2f} °C")
        print(f"总流量: {self.performance['total_flow_rate']:.2f} m³/h")
        print(f"泵功耗: {self.performance['total_power']:.2f} kW")
        print(f"泵扬程: {self.performance['head']:.2f} m")
        print(f"泵效率: {self.performance['efficiency'] * 100:.2f}%\n")