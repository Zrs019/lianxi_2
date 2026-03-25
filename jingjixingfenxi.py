import pandas as pd
import numpy as np
from scipy.optimize import minimize
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# 函数定义
def calculate_cost(x, demand, electricity_price, steam_price):
    """
    计算一天的总运行费用
    :param x: 变量数组，包含冷机、蓄冷和蒸汽制冷的分配量 (x_chiller, x_icing, x_steam)。
    :param demand: 每小时的冷量需求。
    :param electricity_price: 每小时的电价。
    :param steam_price: 蒸汽价格（单位：元/吨）。
    :return 总费用。
    """
    cost_per_hour = []  # 每小时费用
    max_electricity = 0  # 一天中的最大耗电量

    for i in range(len(demand)):
        x_chiller = x[i * 3]
        x_icing = x[i * 3 + 1]
        x_steam = x[i * 3 + 2]

        # 计算每小时耗电量和蒸汽消耗量
        electricity_usage = (x_chiller / 6.26) + (x_icing / 4.24) + 12.65 * (x_steam > 0)
        steam_usage = x_steam / 1.76  # 每小时蒸汽用量
        
        max_electricity = max(max_electricity, electricity_usage)

        # 计算总费用
        electricity_cost = electricity_usage * electricity_price[i]
        total_cost = electricity_cost + (max_electricity * 0.0078) + (steam_usage * steam_price)
        cost_per_hour.append(total_cost)

    return sum(cost_per_hour)

def constraint_total_demand(x, demand):
    """
    确保每小时的冷量分配满足需求
    :param x: 变量数组，包含冷机、蓄冷和蒸汽制冷的分配量。
    :param demand: 每小时的冷量需求。
    :return: 约束条件
    """
    cons = []
    for i in range(len(demand)):
        x_chiller = x[i * 3]
        x_icing = x[i * 3 + 1]
        x_steam = x[i * 3 + 2]
        cons.append(x_chiller + x_icing + x_steam - demand[i])
    return cons

def main():
    # Step 1: 用户选择 Excel 文件并读取数据
    root = Tk()
    root.withdraw()  # 隐藏主窗口
    file_path = askopenfilename(title="选择一个 Excel 文件", filetypes=(("Excel 文件", "*.xlsx"), ("所有文件", "*.*")))
    
    if not file_path:
        print("未选择文件，程序终止")
        return

    data_demand = pd.read_excel(file_path, sheet_name="demand")
    data_chillers = pd.read_excel(file_path, sheet_name="chillers")
    data_icing = pd.read_excel(file_path, sheet_name="icing")
    data_steam = pd.read_excel(file_path, sheet_name="steam")

    # Step 2: 从 Excel 中提取数据
    hours = len(data_demand)
    demand = data_demand['需冷量'].values  # 每小时需冷量
    electricity_price = data_demand['电价'].values  # 每小时电价
    steam_price = 206  # 蒸汽单价（元/吨）

    # 冷机、蓄冷罐和蒸汽的参数
    chiller_capacity = data_chillers.loc[0, "设计冷量kw"]
    icing_capacity = data_icing.loc[0, "额定制冷量kw"]
    steam_capacity = data_steam.loc[0, "设计冷量kw"]

    # Step 3: 初始化优化问题
    x0 = [0] * (hours * 3)  # 冷机、蓄冷、蒸汽每小时使用量的初始值
    bounds = []
    for _ in range(hours):
        bounds.extend([(0, chiller_capacity), (0, icing_capacity), (0, steam_capacity)])
    
    # 添加约束条件：确保冷热量满足需求
    cons = {'type': 'eq', 'fun': constraint_total_demand, 'args': [demand]}
    
    # 优化问题求解
    result = minimize(
        fun=calculate_cost,
        x0=x0,
        args=(demand, electricity_price, steam_price),
        constraints=cons,
        bounds=bounds,
        method='SLSQP'
    )

    # Step 4: 输出结果
    if result.success:
        print("优化成功！最小费用为：", result.fun)
        optimized_schedule = np.array(result.x).reshape((hours, 3))
        output_df = pd.DataFrame(
            optimized_schedule,
            columns=["冷机制冷量", "蓄冷罐放冷量", "蒸汽制冷量"],
            index=data_demand["时间"]
        )
        output_df.to_excel("optimization_result.xlsx", index=True)
        print("优化的运行方案已保存到 'optimization_result.xlsx'")
    else:
        print("优化失败：", result.message)

if __name__ == '__main__':
    main()