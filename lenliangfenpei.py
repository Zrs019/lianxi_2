import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog, messagebox

# 固定负荷档位
LOAD_LEVELS = [0.25, 0.50, 0.75, 1.00]


def choose_load_for_demand(capacity, demand):
    """
    选择能够覆盖 demand 的最小负荷档位
    - demand <= 0 -> 返回 0
    - 若 100% 仍无法覆盖，则返回 1.0
    """
    if demand <= 0:
        return 0.0

    for ratio in LOAD_LEVELS:
        if capacity * ratio >= demand:
            return ratio

    return 1.0


def best_cop_at_possible_load(chiller, demand):
    """
    计算某台冷机在承担 demand 时，所需最小档位及其对应 COP
    """
    ratio = choose_load_for_demand(chiller["capacity"], demand)
    if ratio == 0:
        return 0.0, 0.0
    return ratio, chiller["cop"][ratio]


def calc_power(actual_cooling, cop):
    """
    功率 = 实际供冷量 / COP
    """
    if cop is None or cop == 0:
        return 0.0
    return actual_cooling / cop


def dispatch_two_chillers(demand, ch1, ch2):
    """
    分配逻辑：
    1. 比较两台冷机在当前需求下的优先 COP
    2. 优先让 COP 更高的冷机先承担
    3. 剩余需求给另一台冷机
    4. 负荷档位用于确定 COP
    5. 实际供冷量按实际承担需求记，不按档位额定冷量记
    """

    total_capacity = ch1["capacity"] + ch2["capacity"]

    if demand > total_capacity:
        ratio1, cop1 = best_cop_at_possible_load(ch1, ch1["capacity"])
        ratio2, cop2 = best_cop_at_possible_load(ch2, ch2["capacity"])

        if cop1 >= cop2:
            first, second = ch1, ch2
            first_id, second_id = 1, 2
        else:
            first, second = ch2, ch1
            first_id, second_id = 2, 1

        first_ratio = 1.0
        first_actual = first["capacity"]
        first_cop = first["cop"][1.0]
        first_power = calc_power(first_actual, first_cop)

        second_ratio = 1.0
        second_actual = second["capacity"]
        second_cop = second["cop"][1.0]
        second_power = calc_power(second_actual, second_cop)

        result = {
            "需求": demand,
            "状态": "需求超过两台冷机总容量，无法完全满足",
            "机组1负荷率": 0.0,
            "机组1负荷档位": "0%",
            "机组1实际供冷量": 0.0,
            "机组1COP": 0.0,
            "机组1功率": 0.0,
            "机组2负荷率": 0.0,
            "机组2负荷档位": "0%",
            "机组2实际供冷量": 0.0,
            "机组2COP": 0.0,
            "机组2功率": 0.0,
            "总功率": 0.0,
            "未满足冷量": demand - total_capacity
        }

        result[f"机组{first_id}负荷率"] = first_ratio
        result[f"机组{first_id}负荷档位"] = "100%"
        result[f"机组{first_id}实际供冷量"] = first_actual
        result[f"机组{first_id}COP"] = first_cop
        result[f"机组{first_id}功率"] = first_power

        result[f"机组{second_id}负荷率"] = second_ratio
        result[f"机组{second_id}负荷档位"] = "100%"
        result[f"机组{second_id}实际供冷量"] = second_actual
        result[f"机组{second_id}COP"] = second_cop
        result[f"机组{second_id}功率"] = second_power

        result["总功率"] = first_power + second_power
        return result

    ratio1, cop1 = best_cop_at_possible_load(ch1, min(demand, ch1["capacity"]))
    ratio2, cop2 = best_cop_at_possible_load(ch2, min(demand, ch2["capacity"]))

    if cop1 >= cop2:
        first, second = ch1, ch2
        first_id, second_id = 1, 2
    else:
        first, second = ch2, ch1
        first_id, second_id = 2, 1

    first_actual = min(demand, first["capacity"])
    first_ratio = choose_load_for_demand(first["capacity"], first_actual)
    first_cop = first["cop"][first_ratio] if first_ratio > 0 else 0.0
    first_power = calc_power(first_actual, first_cop)

    remaining = max(0.0, demand - first_actual)

    if remaining > 0:
        second_actual = remaining
        second_ratio = choose_load_for_demand(second["capacity"], second_actual)
        second_cop = second["cop"][second_ratio]
        second_power = calc_power(second_actual, second_cop)
    else:
        second_actual = 0.0
        second_ratio = 0.0
        second_cop = 0.0
        second_power = 0.0

    result = {
        "需求": demand,
        "状态": "满足",
        "机组1负荷率": 0.0,
        "机组1负荷档位": "0%",
        "机组1实际供冷量": 0.0,
        "机组1COP": 0.0,
        "机组1功率": 0.0,
        "机组2负荷率": 0.0,
        "机组2负荷档位": "0%",
        "机组2实际供冷量": 0.0,
        "机组2COP": 0.0,
        "机组2功率": 0.0,
        "总功率": 0.0,
        "未满足冷量": 0.0
    }

    result[f"机组{first_id}负荷率"] = first_ratio
    result[f"机组{first_id}负荷档位"] = f"{int(first_ratio * 100)}%"
    result[f"机组{first_id}实际供冷量"] = first_actual
    result[f"机组{first_id}COP"] = first_cop
    result[f"机组{first_id}功率"] = first_power

    result[f"机组{second_id}负荷率"] = second_ratio
    result[f"机组{second_id}负荷档位"] = f"{int(second_ratio * 100)}%"
    result[f"机组{second_id}实际供冷量"] = second_actual
    result[f"机组{second_id}COP"] = second_cop
    result[f"机组{second_id}功率"] = second_power

    result["总功率"] = first_power + second_power

    return result


def read_input_excel(file_path):
    """
    读取本地 Excel
    Sheet1: demand
    Sheet2: chillers
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到文件: {file_path}")

    demand_df = pd.read_excel(file_path, sheet_name="demand")
    chiller_df = pd.read_excel(file_path, sheet_name="chillers")

    required_demand_cols = ["时间", "冷量需求"]
    for col in required_demand_cols:
        if col not in demand_df.columns:
            raise ValueError(f"demand sheet 缺少必要列: {col}")

    required_chiller_cols = ["冷机", "设计冷量", "COP_100", "COP_75", "COP_50", "COP_25"]
    for col in required_chiller_cols:
        if col not in chiller_df.columns:
            raise ValueError(f"chillers sheet 缺少必要列: {col}")

    if len(chiller_df) != 2:
        raise ValueError("当前程序要求 chillers sheet 中必须正好有两台冷机")

    chillers = []
    for _, row in chiller_df.iterrows():
        chillers.append({
            "name": row["冷机"],
            "capacity": float(row["设计冷量"]),
            "cop": {
                1.00: float(row["COP_100"]),
                0.75: float(row["COP_75"]),
                0.50: float(row["COP_50"]),
                0.25: float(row["COP_25"])
            }
        })

    return demand_df, chillers


def select_input_file():
    """
    弹窗选择 Excel 文件
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    file_path = filedialog.askopenfilename(
        title="请选择输入Excel文件",
        filetypes=[("Excel 文件", "*.xlsx *.xls")]
    )

    root.destroy()
    return file_path


def build_output_path(input_file):
    """
    输出文件保存到输入文件同目录，命名为 原文件名_output.xlsx
    例如：
    D:/data/input.xlsx -> D:/data/input_output.xlsx
    """
    folder = os.path.dirname(input_file)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(folder, f"{base_name}_output.xlsx")
    return output_file


def main():
    try:
        input_file = select_input_file()

        if not input_file:
            print("未选择文件，程序已退出。")
            return

        demand_df, chillers = read_input_excel(input_file)
        ch1, ch2 = chillers[0], chillers[1]

        results = []

        for _, row in demand_df.iterrows():
            time_value = row["时间"]
            demand = float(row["冷量需求"])

            res = dispatch_two_chillers(demand, ch1, ch2)
            res["时间"] = time_value
            results.append(res)

        result_df = pd.DataFrame(results)

        result_df = result_df[
            [
                "时间",
                "需求",
                "状态",
                "机组1负荷率",
                "机组1负荷档位",
                "机组1实际供冷量",
                "机组1COP",
                "机组1功率",
                "机组2负荷率",
                "机组2负荷档位",
                "机组2实际供冷量",
                "机组2COP",
                "机组2功率",
                "总功率",
                "未满足冷量"
            ]
        ]

        output_file = build_output_path(input_file)
        result_df.to_excel(output_file, index=False)

        print(f"计算完成，结果已保存到: {output_file}")

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        messagebox.showinfo("完成", f"计算完成，结果已保存到：\n{output_file}")
        root.destroy()

    except Exception as e:
        print(f"程序运行出错：{e}")
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        messagebox.showerror("错误", f"程序运行出错：\n{e}")
        root.destroy()


if __name__ == "__main__":
    main()