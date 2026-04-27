# -*- coding: utf-8 -*-
# pip install pandas pulp openpyxl

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import pulp

# =========================
# 1) 读Excel
# =========================
def load_input_excel(path):
    """
    约定工作表：
    1) 负荷: 列[hour, load_kwc]
    2) 电制冷机: 列[name, Qmax, COP_100, COP_75, COP_50, COP_25]
    3) 蓄冰机: 列[name, Q_charge_max, P_charge_max, COP, E_ice_max, discharge_ratio]
    4) 电价: 列[hour, price]
    5) 参数(可选): 列[key, value]，如 pf=0.85, demand_price=48, days_in_month=30, p_base=0
    """
    xls = pd.ExcelFile(path)

    df_load = pd.read_excel(xls, "负荷")
    df_chiller = pd.read_excel(xls, "电制冷机")
    df_ice = pd.read_excel(xls, "蓄冰机")
    df_price = pd.read_excel(xls, "电价")

    params = {"pf": 0.85, "demand_price": 48.0, "days_in_month": 30, "p_base": 0.0}
    if "参数" in xls.sheet_names:
        df_param = pd.read_excel(xls, "参数")
        for _, r in df_param.iterrows():
            k = str(r["key"]).strip()
            v = float(r["value"])
            params[k] = v

    # 负荷、价格转字典
    q_load = {int(r["hour"]): float(r["load_kwc"]) for _, r in df_load.iterrows()}
    price = {int(r["hour"]): float(r["price"]) for _, r in df_price.iterrows()}

    # 电制冷机
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

    # 蓄冰机
    ice_units = {}
    for _, r in df_ice.iterrows():
        name = str(r["name"]).strip()
        E_ice_max = float(r["E_ice_max"])
        discharge_ratio = float(r["discharge_ratio"])
        ice_units[name] = {
            "Q_charge_max": float(r["Q_charge_max"]),
            "P_charge_max": float(r["P_charge_max"]),
            "COP": float(r["COP"]),
            "E_ice_max": E_ice_max,
            "Q_discharge_max": discharge_ratio * E_ice_max
        }

    return q_load, price, chillers, ice_units, params


# =========================
# 2) 机组选择弹窗
# =========================
# ---------- 1) 设备选择：蓄冰机改多选 ----------
def select_units_gui(chillers, ice_units):
    selected = {"chillers": [], "ice_units": []}

    import tkinter as tk
    from tkinter import messagebox

    win = tk.Tk()
    win.title("设备选择")

    tk.Label(win, text="选择电制冷机（可多选）").grid(row=0, column=0, sticky="w", padx=10, pady=5)
    ch_vars = {}
    row = 1
    for c in chillers.keys():
        v = tk.BooleanVar(value=True)
        ch_vars[c] = v
        tk.Checkbutton(win, text=c, variable=v).grid(row=row, column=0, sticky="w", padx=20)
        row += 1

    tk.Label(win, text="选择蓄冰机（可多选）").grid(row=0, column=1, sticky="w", padx=10, pady=5)
    ice_vars = {}
    row2 = 1
    for i in ice_units.keys():
        v = tk.BooleanVar(value=True)
        ice_vars[i] = v
        tk.Checkbutton(win, text=i, variable=v).grid(row=row2, column=1, sticky="w", padx=20)
        row2 += 1

    def on_ok():
        selected["chillers"] = [k for k, v in ch_vars.items() if v.get()]
        selected["ice_units"] = [k for k, v in ice_vars.items() if v.get()]
        if not selected["chillers"]:
            messagebox.showerror("错误", "至少选择一台电制冷机")
            return
        if not selected["ice_units"]:
            messagebox.showerror("错误", "至少选择一台蓄冰机")
            return
        win.destroy()

    tk.Button(win, text="确定", command=on_ok).grid(row=max(row, row2)+1, column=0, columnspan=2, pady=10)
    win.mainloop()
    return selected


# ---------- 2) 多台蓄冰机汇总 ----------
def aggregate_ice_units(ice_units_all, selected_names):
    """
    把多台蓄冰机等效汇总成一台“虚拟蓄冰机”
    """
    agg = {
        "Q_charge_max": 0.0,   # kW冷
        "P_charge_max": 0.0,   # kW电
        "E_ice_max": 0.0,      # kWh冷
        "Q_discharge_max": 0.0 # kW冷
    }

    # COP按总制冷/总功率折算（加权）
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
# 3) 优化核心
# =========================
def optimize_dispatch(q_load, price, chillers_all, ice_unit, params, p_cap=None):
    hours = sorted(q_load.keys())
    dt = 1.0

    pf = float(params.get("pf", 0.85))
    demand_price = float(params.get("demand_price", 48.0))
    days_in_month = float(params.get("days_in_month", 30))
    p_base = float(params.get("p_base", 0.0))
    P_base = {h: p_base for h in hours}

    # 分段
    for c, d in chillers_all.items():
        d["seg_cop"] = [d["cop_25"], d["cop_50"], d["cop_75"], d["cop_100"]]

    m = pulp.LpProblem("Ice_Storage_Optimization", pulp.LpMinimize)

    q_ch = {(c, h): pulp.LpVariable(f"q_{c}_{h}", lowBound=0) for c in chillers_all for h in hours}
    p_ch = {(c, h): pulp.LpVariable(f"p_{c}_{h}", lowBound=0) for c in chillers_all for h in hours}
    q_seg = {(c, h, s): pulp.LpVariable(f"qseg_{c}_{h}_{s}", lowBound=0)
             for c in chillers_all for h in hours for s in range(4)}

    q_ice_ch = {h: pulp.LpVariable(f"q_ice_ch_{h}", lowBound=0, upBound=ice_unit["Q_charge_max"]) for h in hours}
    p_ice_ch = {h: pulp.LpVariable(f"p_ice_ch_{h}", lowBound=0, upBound=ice_unit["P_charge_max"]) for h in hours}
    q_dis = {h: pulp.LpVariable(f"q_dis_{h}", lowBound=0, upBound=ice_unit["Q_discharge_max"]) for h in hours}
    soc = {h: pulp.LpVariable(f"soc_{h}", lowBound=0, upBound=ice_unit["E_ice_max"]) for h in hours}

    p_total = {h: pulp.LpVariable(f"p_total_{h}", lowBound=0) for h in hours}
    p_peak = pulp.LpVariable("p_peak", lowBound=0)

    # 机组约束
    for c, d in chillers_all.items():
        Qmax = d["Qmax"]
        seg_cap = [0.25 * Qmax] * 4
        for h in hours:
            m += q_ch[(c, h)] == pulp.lpSum(q_seg[(c, h, s)] for s in range(4))
            for s in range(4):
                m += q_seg[(c, h, s)] <= seg_cap[s]
            m += p_ch[(c, h)] == pulp.lpSum(q_seg[(c, h, s)] / d["seg_cop"][s] for s in range(4))

    # 蓄冰机
    for h in hours:
        m += q_ice_ch[h] == ice_unit["COP"] * p_ice_ch[h]

    # 冷量平衡
    for h in hours:
        m += pulp.lpSum(q_ch[(c, h)] for c in chillers_all) + q_dis[h] == q_load[h] + q_ice_ch[h]

    # SOC（周期）
    for i, h in enumerate(hours):
        if i == 0:
            m += soc[h] == soc[hours[-1]] + q_ice_ch[h] * dt - q_dis[h] * dt
        else:
            hp = hours[i - 1]
            m += soc[h] == soc[hp] + q_ice_ch[h] * dt - q_dis[h] * dt

    # 功率和峰值
    for h in hours:
        m += p_total[h] == P_base[h] + pulp.lpSum(p_ch[(c, h)] for c in chillers_all) + p_ice_ch[h]
        m += p_peak >= p_total[h]
        if p_cap is not None:
            m += p_total[h] <= p_cap

    daily_demand_factor = (demand_price / pf) / days_in_month
    energy_cost_day = pulp.lpSum(price[h] * p_total[h] * dt for h in hours)
    demand_cost_day = daily_demand_factor * p_peak
    m += energy_cost_day + demand_cost_day

    solver = pulp.PULP_CBC_CMD(msg=False)
    m.solve(solver)

    if pulp.LpStatus[m.status] != "Optimal":
        raise RuntimeError(f"优化失败: {pulp.LpStatus[m.status]}")

    rows = []
    for h in hours:
        row = {
            "hour": h,
            "Q_load_kWc": q_load[h],
            "Q_dis_kWc": q_dis[h].value(),
            "Q_ice_ch_kWc": q_ice_ch[h].value(),
            "P_ice_ch_kW": p_ice_ch[h].value(),
            "SOC_kWhc": soc[h].value(),
            "P_total_kW": p_total[h].value(),
            "price": price[h],
            "cost_hour_yuan": price[h] * p_total[h].value() * dt
        }
        for c in chillers_all:
            row[f"Q_{c}_kWc"] = q_ch[(c, h)].value()
            row[f"P_{c}_kW"] = p_ch[(c, h)].value()
        rows.append(row)

    df = pd.DataFrame(rows)

    energy_day = df["cost_hour_yuan"].sum()
    peak_kw = p_peak.value()
    demand_day = daily_demand_factor * peak_kw
    total_day = energy_day + demand_day

    energy_month = energy_day * days_in_month
    demand_month = (demand_price / pf) * peak_kw
    total_month = energy_month + demand_month

    summary = pd.DataFrame([{
        "status": pulp.LpStatus[m.status],
        "energy_cost_day": energy_day,
        "demand_cost_day": demand_day,
        "total_cost_day": total_day,
        "peak_kw": peak_kw,
        "energy_cost_month": energy_month,
        "demand_cost_month": demand_month,
        "total_cost_month": total_month
    }])

    return df, summary


# =========================
# 4) 主程序
# =========================
def main():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="选择输入Excel",
        filetypes=[("Excel files", "*.xlsx *.xls")]
    )
    if not file_path:
        print("未选择文件，退出。")
        return

    q_load, price, chillers, ice_units, params = load_input_excel(file_path)

    # 选择设备

    sel = select_units_gui(chillers, ice_units)
    chosen_chillers = {k: chillers[k] for k in sel["chillers"]}
    chosen_ice = aggregate_ice_units(ice_units, sel["ice_units"])
    # 可选：输入需量红线
    pcap = None
    use_cap = messagebox.askyesno("需量红线", "是否设置需量红线P_cap？")
    if use_cap:
        cap_win = tk.Tk()
        cap_win.title("输入P_cap(kW)")
        tk.Label(cap_win, text="请输入P_cap(kW):").pack(padx=10, pady=5)
        cap_var = tk.StringVar(value="2500")
        tk.Entry(cap_win, textvariable=cap_var).pack(padx=10, pady=5)

        holder = {"val": None}
        def ok_cap():
            try:
                holder["val"] = float(cap_var.get())
                cap_win.destroy()
            except:
                messagebox.showerror("错误", "请输入数字")
        tk.Button(cap_win, text="确定", command=ok_cap).pack(pady=8)
        cap_win.mainloop()
        pcap = holder["val"]

    # 优化
    df, summary = optimize_dispatch(q_load, price, chosen_chillers, chosen_ice, params, p_cap=pcap)

    # 导出
    out_path = filedialog.asksaveasfilename(
        title="保存结果Excel",
        defaultextension=".xlsx",
        filetypes=[("Excel files", "*.xlsx")]
    )
    if not out_path:
        out_path = "dispatch_result_selected.xlsx"

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        df.to_excel(writer, sheet_name="dispatch", index=False)

    messagebox.showinfo("完成", f"优化完成，结果已保存：\n{out_path}")
    print("Done:", out_path)


if __name__ == "__main__":
    main()