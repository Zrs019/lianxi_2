# -*- coding: utf-8 -*-
# pip install pandas pulp openpyxl
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import pandas as pd
import pulp
import datetime as dt


# =========================
# 通用解析
# =========================
def parse_hour(v):
    """兼容 hour: int/float, datetime.time, Timestamp, '0:00' 等"""
    if pd.isna(v):
        raise ValueError("hour 存在空值")

    if isinstance(v, (int, float)):
        h = int(v)
        if 0 <= h <= 23:
            return h
        raise ValueError(f"非法小时数: {v}")

    if isinstance(v, dt.time):
        return int(v.hour)

    if isinstance(v, (pd.Timestamp, dt.datetime)):
        return int(v.hour)

    s = str(v).strip()
    if ":" in s:
        h = int(s.split(":")[0])
        if 0 <= h <= 23:
            return h
    else:
        try:
            h = int(float(s))
            if 0 <= h <= 23:
                return h
        except Exception:
            pass

    raise ValueError(f"无法解析hour值: {v} (type={type(v)})")


# =========================
# 文件对话框（单root稳定）
# =========================
def ask_input_file(root):
    root.lift()
    root.attributes("-topmost", True)
    root.update()
    path = filedialog.askopenfilename(
        parent=root,
        title="选择输入Excel",
        filetypes=[("Excel files", "*.xlsx *.xls")]
    )
    root.attributes("-topmost", False)
    return path


def ask_save_file(root):
    root.lift()
    root.attributes("-topmost", True)
    root.update()
    path = filedialog.asksaveasfilename(
        parent=root,
        title="保存结果Excel",
        defaultextension=".xlsx",
        filetypes=[("Excel files", "*.xlsx")]
    )
    root.attributes("-topmost", False)
    return path


# =========================
# 读取输入Excel
# =========================
def load_input_excel(path):
    """
    Excel约定：
    - 负荷: [hour, load_kwc] 或 [hour, load_k] 或 [hour, load]
    - 电制冷机: [name, Qmax, COP_100, COP_75, COP_50, COP_25]
    - 蓄冰机: [name, Q_charge_max, P_charge_max, COP, E_ice_max, discharge_ratio]
    - 电价: [hour, price]
    - 参数(可选): [key, value], 如 pf=0.85, demand_price=48, days_in_month=30, p_base=0
    """
    xls = pd.ExcelFile(path)

    required_sheets = ["负荷", "电制冷机", "蓄冰机", "电价"]
    for s in required_sheets:
        if s not in xls.sheet_names:
            raise ValueError(f"缺少工作表: {s}")

    df_load = pd.read_excel(xls, "负荷")
    df_chiller = pd.read_excel(xls, "电制冷机")
    df_ice = pd.read_excel(xls, "蓄冰机")
    df_price = pd.read_excel(xls, "电价")

    # 负荷列名兼容
    load_col = None
    for c in ["load_kwc", "load_k", "load"]:
        if c in df_load.columns:
            load_col = c
            break
    if load_col is None:
        raise ValueError(f"负荷表缺少负荷列(load_kwc/load_k/load)，当前列: {list(df_load.columns)}")

    params = {"pf": 0.85, "demand_price": 48.0, "days_in_month": 30, "p_base": 0.0}
    if "参数" in xls.sheet_names:
        df_param = pd.read_excel(xls, "参数")
        for _, r in df_param.iterrows():
            k = str(r["key"]).strip()
            v = float(r["value"])
            params[k] = v

    q_load = {}
    for _, r in df_load.iterrows():
        h = parse_hour(r["hour"])
        q_load[h] = float(r[load_col])

    price = {}
    for _, r in df_price.iterrows():
        h = parse_hour(r["hour"])
        price[h] = float(r["price"])

    missing_load = [h for h in range(24) if h not in q_load]
    missing_price = [h for h in range(24) if h not in price]
    if missing_load:
        raise ValueError(f"负荷缺少小时: {missing_load}")
    if missing_price:
        raise ValueError(f"电价缺少小时: {missing_price}")

    chillers = {}
    for _, r in df_chiller.iterrows():
        name = str(r["name"]).strip()
        chillers[name] = {
            "Qmax": float(r["Qmax"]),
            "cop_100": float(r["COP_100"]),
            "cop_75": float(r["COP_75"]),
            "cop_50": float(r["COP_50"]),
            "cop_25": float(r["COP_25"]),
        }

    ice_units = {}
    for _, r in df_ice.iterrows():
        name = str(r["name"]).strip()
        E_ice_max = float(r["E_ice_max"])
        discharge_ratio = float(r["discharge_ratio"])
        ice_units[name] = {
            "Q_charge_max": float(r["Q_charge_max"]),   # kW冷（蓄冰机制冷量）
            "P_charge_max": float(r["P_charge_max"]),   # kW电（蓄冰机功率）
            "COP": float(r["COP"]),
            "E_ice_max": E_ice_max,                     # kWh冷
            "Q_discharge_max": discharge_ratio * E_ice_max  # kW冷
        }

    return q_load, price, chillers, ice_units, params


# =========================
# 设备选择弹窗（单root + 模态）
# =========================
class UnitSelectDialog(simpledialog.Dialog):
    def __init__(self, parent, chillers, ice_units, title="设备选择"):
        self.chillers = chillers
        self.ice_units = ice_units
        self.result_data = {"chillers": [], "ice_units": []}
        super().__init__(parent, title)

    def body(self, master):
        self.resizable(False, False)

        tk.Label(master, text="选择电制冷机（可多选）").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.ch_vars = {}
        r = 1
        for c in self.chillers.keys():
            v = tk.BooleanVar(value=True)
            self.ch_vars[c] = v
            tk.Checkbutton(master, text=c, variable=v).grid(row=r, column=0, sticky="w", padx=20)
            r += 1

        tk.Label(master, text="选择蓄冰机（可多选）").grid(row=0, column=1, sticky="w", padx=10, pady=5)
        self.ice_vars = {}
        r2 = 1
        for i in self.ice_units.keys():
            v = tk.BooleanVar(value=True)
            self.ice_vars[i] = v
            tk.Checkbutton(master, text=i, variable=v).grid(row=r2, column=1, sticky="w", padx=20)
            r2 += 1

        return master

    def validate(self):
        ch = [k for k, v in self.ch_vars.items() if v.get()]
        ice = [k for k, v in self.ice_vars.items() if v.get()]
        if not ch:
            messagebox.showerror("错误", "至少选择一台电制冷机", parent=self)
            return False
        if not ice:
            messagebox.showerror("错误", "至少选择一台蓄冰机", parent=self)
            return False
        self.result_data = {"chillers": ch, "ice_units": ice}
        return True

    def apply(self):
        pass


def select_units_gui(root, chillers, ice_units):
    root.deiconify()
    root.lift()
    root.attributes("-topmost", True)
    root.update_idletasks()
    root.attributes("-topmost", False)

    dlg = UnitSelectDialog(root, chillers, ice_units, title="设备选择")
    return dlg.result_data


def aggregate_ice_units(ice_units_all, selected_names):
    """多台蓄冰机汇总成一台等效蓄冰系统"""
    agg = {
        "Q_charge_max": 0.0,
        "P_charge_max": 0.0,
        "E_ice_max": 0.0,
        "Q_discharge_max": 0.0
    }
    total_q = 0.0
    total_p = 0.0

    for n in selected_names:
        u = ice_units_all[n]
        agg["Q_charge_max"] += float(u["Q_charge_max"])
        agg["P_charge_max"] += float(u["P_charge_max"])
        agg["E_ice_max"] += float(u["E_ice_max"])
        agg["Q_discharge_max"] += float(u["Q_discharge_max"])
        total_q += float(u["Q_charge_max"])
        total_p += float(u["P_charge_max"])

    agg["COP"] = (total_q / total_p) if total_p > 1e-9 else 0.0
    return agg


# =========================
# 优化模型（逻辑修正版）
# =========================
def optimize_dispatch(q_load, price, chillers_all, ice_unit, params):
    """
    关键物理逻辑：
    1) 建筑负荷平衡：sum(Q_chiller) + Q_dis = Q_load
    2) 蓄冰机制冰不进入建筑负荷平衡，只进入SOC：
       SOC_t = SOC_{t-1} + Q_ice_ch - Q_dis
    """
    hours = sorted(q_load.keys())
    dt = 1.0

    pf = float(params.get("pf", 0.85))
    demand_price = float(params.get("demand_price", 48.0))
    days_in_month = float(params.get("days_in_month", 30))
    p_base = float(params.get("p_base", 0.0))
    P_base = {h: p_base for h in hours}

    # 分段COP准备
    for c, d in chillers_all.items():
        d["seg_cop"] = [d["cop_25"], d["cop_50"], d["cop_75"], d["cop_100"]]

    m = pulp.LpProblem("Ice_Storage_Optimization", pulp.LpMinimize)

    # 变量
    q_ch = {(c, h): pulp.LpVariable(f"q_{c}_{h}", lowBound=0) for c in chillers_all for h in hours}
    p_ch = {(c, h): pulp.LpVariable(f"p_{c}_{h}", lowBound=0) for c in chillers_all for h in hours}
    q_seg = {(c, h, s): pulp.LpVariable(f"qseg_{c}_{h}_{s}", lowBound=0)
             for c in chillers_all for h in hours for s in range(4)}

    # 蓄冰机变量
    q_ice_ch = {h: pulp.LpVariable(f"q_ice_ch_{h}", lowBound=0, upBound=ice_unit["Q_charge_max"]) for h in hours}
    p_ice_ch = {h: pulp.LpVariable(f"p_ice_ch_{h}", lowBound=0, upBound=ice_unit["P_charge_max"]) for h in hours}
    q_dis = {h: pulp.LpVariable(f"q_dis_{h}", lowBound=0, upBound=min(ice_unit["Q_discharge_max"], q_load[h])) for h in hours}
    soc = {h: pulp.LpVariable(f"soc_{h}", lowBound=0, upBound=ice_unit["E_ice_max"]) for h in hours}

    # 电功率变量
    p_total = {h: pulp.LpVariable(f"p_total_{h}", lowBound=0) for h in hours}
    p_peak = pulp.LpVariable("p_peak", lowBound=0)

    # 电制冷机分段约束
    for c, d in chillers_all.items():
        Qmax = d["Qmax"]
        seg_cap = [0.25 * Qmax] * 4
        for h in hours:
            m += q_ch[(c, h)] == pulp.lpSum(q_seg[(c, h, s)] for s in range(4))
            for s in range(4):
                m += q_seg[(c, h, s)] <= seg_cap[s]
            m += p_ch[(c, h)] == pulp.lpSum(q_seg[(c, h, s)] / d["seg_cop"][s] for s in range(4))

    # 逐时约束
    valley_hours = [0, 1, 2, 3, 4, 5, 6, 7]

    for h in hours:
        # 蓄冰机功率-冷量关系（蓄冰机独立制冰）
        if h in valley_hours:
            m += q_ice_ch[h] == ice_unit["COP"] * p_ice_ch[h]
        else:
            m += q_ice_ch[h] == 0
            m += p_ice_ch[h] == 0

        # 建筑负荷平衡（关键修正：不再含 q_ice_ch）
        m += pulp.lpSum(q_ch[(c, h)] for c in chillers_all) + q_dis[h] == q_load[h]

        # 总功率
        m += p_total[h] == P_base[h] + pulp.lpSum(p_ch[(c, h)] for c in chillers_all) + p_ice_ch[h]
        m += p_peak >= p_total[h]

    # SOC能量守恒
    soc_init = 0.5 * ice_unit["E_ice_max"]  # 可改为参数输入
    m += soc[0] == soc_init + q_ice_ch[0] * dt - q_dis[0] * dt

    for i in range(1, len(hours)):
        h = hours[i]
        hp = hours[i - 1]
        m += soc[h] == soc[hp] + q_ice_ch[h] * dt - q_dis[h] * dt

    # 日循环：日末=日初，避免“透支或无限囤冰”
    m += soc[23] == soc_init

    # 目标函数（只设置一次）
    daily_demand_factor = (demand_price / pf) / days_in_month
    energy_cost_day = pulp.lpSum(price[h] * p_total[h] * dt for h in hours)
    demand_cost_day = daily_demand_factor * p_peak
    eps_penalty = 1e-4  # 微小惩罚，减少无意义充放循环
    m += energy_cost_day + demand_cost_day + eps_penalty * pulp.lpSum(q_dis[h] + q_ice_ch[h] for h in hours)

    # 求解器
    solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=60, gapRel=0.01, threads=0)
    m.solve(solver)
    status = pulp.LpStatus[m.status]

    # 结果表
    rows = []
    for h in hours:
        q_ch_sum = sum((q_ch[(c, h)].value() or 0.0) for c in chillers_all)
        row = {
            "hour": h,
            "Q_load_kWc": q_load[h],
            "Q_ch_total_kWc": q_ch_sum,
            "Q_dis_kWc": q_dis[h].value(),
            "Q_ice_ch_kWc": q_ice_ch[h].value(),
            "P_ice_ch_kW": p_ice_ch[h].value(),
            "SOC_kWhc": soc[h].value(),
            "P_total_kW": p_total[h].value(),
            "price": price[h],
            "cost_hour_yuan": price[h] * (p_total[h].value() or 0) * dt,
            # 平衡误差自检：应接近0
            "Q_balance_error": q_ch_sum + (q_dis[h].value() or 0) - q_load[h]
        }
        for c in chillers_all:
            row[f"Q_{c}_kWc"] = q_ch[(c, h)].value()
            row[f"P_{c}_kW"] = p_ch[(c, h)].value()
        rows.append(row)

    df = pd.DataFrame(rows)

    energy_day = df["cost_hour_yuan"].sum()
    peak_kw = p_peak.value() or 0
    demand_day = daily_demand_factor * peak_kw
    total_day = energy_day + demand_day

    energy_month = energy_day * days_in_month
    demand_month = (demand_price / pf) * peak_kw
    total_month = energy_month + demand_month

    eps = 1e-6
    charge_hours = int((df["P_ice_ch_kW"] > eps).sum())
    avg_charge_power = float(df.loc[df["P_ice_ch_kW"] > eps, "P_ice_ch_kW"].mean()) if charge_hours > 0 else 0.0
    max_charge_power = float(df["P_ice_ch_kW"].max())

    summary = pd.DataFrame([{
        "status": status,
        "energy_cost_day": energy_day,
        "demand_cost_day": demand_day,
        "total_cost_day": total_day,
        "peak_kw": peak_kw,
        "energy_cost_month": energy_month,
        "demand_cost_month": demand_month,
        "total_cost_month": total_month,
        "charge_hours_day": charge_hours,
        "avg_charge_power_kw": avg_charge_power,
        "max_charge_power_kw": max_charge_power,
        "max_abs_balance_error": df["Q_balance_error"].abs().max()
    }])

    return df, summary


# =========================
# 主程序（单root）
# =========================
def main():
    root = tk.Tk()
    root.withdraw()

    try:
        # 1) 选输入文件
        file_path = ask_input_file(root)
        if not file_path:
            return

        root.deiconify()
        root.withdraw()

        # 2) 读数据
        q_load, price, chillers, ice_units, params = load_input_excel(file_path)

        # 3) 选设备
        sel = select_units_gui(root, chillers, ice_units)
        chosen_chillers = {k: chillers[k] for k in sel["chillers"]}
        chosen_ice = aggregate_ice_units(ice_units, sel["ice_units"])

        # 4) 优化
        df, summary = optimize_dispatch(q_load, price, chosen_chillers, chosen_ice, params)

        # 5) 保存结果
        out_path = ask_save_file(root)
        if not out_path:
            out_path = "dispatch_result_selected.xlsx"

        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            summary.to_excel(writer, sheet_name="summary", index=False)
            df.to_excel(writer, sheet_name="dispatch", index=False)

        messagebox.showinfo("完成", f"优化完成，结果已保存：\n{out_path}", parent=root)
        print("Done:", out_path)
        print(summary.to_string(index=False))

    except Exception as e:
        messagebox.showerror("错误", str(e), parent=root)
        raise
    finally:
        root.destroy()


if __name__ == "__main__":
    main()