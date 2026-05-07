import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False


CP_WATER_KJ_PER_KG_K = 4.186
USERS = (3, 4, 6)
SUPPLY_TEMP_SET_C = 7.0
SUPPLY_TEMP_WARN_C = 12.0
RETURN_TEMP_MAX_C = 20.0


def load_series(mat_data, name):
    if name not in mat_data:
        raise KeyError(f"sim_result.mat 中缺少变量: {name}")
    return np.asarray(mat_data[name], dtype=float).squeeze()


def load_optional_series(mat_data, names):
    for name in names:
        if name in mat_data:
            return np.asarray(mat_data[name], dtype=float).squeeze(), name
    return None, None


def maybe_kelvin_to_celsius(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size and np.nanmedian(finite) > 200:
        return values - 273.15
    return values


def hourly_average(values, time_sec, total_hours):
    values = np.asarray(values, dtype=float).squeeze()
    time_sec = np.asarray(time_sec, dtype=float).squeeze()
    hourly = np.full(total_hours, np.nan)

    if values.size == 0:
        return hourly

    # Simulink 保存的 tout 有时和各 To Workspace 信号采样长度不同。
    # 长度不一致时，按信号自身长度在完整仿真时长内均匀展开，避免布尔索引错位。
    if time_sec.size != values.size:
        if time_sec.size > 1 and np.isfinite(time_sec).any():
            end_time = float(np.nanmax(time_sec))
        else:
            end_time = float(total_hours * 3600.0)
        if values.size == 1:
            return np.full(total_hours, float(values[0]))
        time_sec = np.linspace(0.0, end_time, values.size)

    for hour in range(total_hours):
        start = hour * 3600.0
        end = (hour + 1) * 3600.0
        mask = (time_sec >= start) & (time_sec < end)
        if np.any(mask):
            hourly[hour] = np.nanmean(values[mask])
        elif time_sec.size == values.size and time_sec.size > 1:
            hourly[hour] = np.interp(start + 1800.0, time_sec, values)

    return hourly


def load_demand_kw(boundary_file, total_hours):
    path = Path(boundary_file)
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    demand = {}
    for user in USERS:
        col = f"user{user}_demand_kw"
        heat_col = f"user{user}_heat"
        if col in df:
            demand[user] = df[col].to_numpy(dtype=float)[:total_hours]
        elif heat_col in df:
            demand[user] = df[heat_col].to_numpy(dtype=float)[:total_hours] / 1000.0
    return demand


def build_hourly_dataframe(mat_file, boundary_file):
    try:
        mat_data = sio.loadmat(mat_file, squeeze_me=True, struct_as_record=False)
    except NotImplementedError as exc:
        raise RuntimeError(
            f"{mat_file} 是 MATLAB -v7.3/HDF5 格式，当前 Python 环境不能直接读取。"
            "请在 MATLAB 中重新运行 save_sim_result.m；脚本已改为 -v7 保存，"
            "重新保存后再运行 plot_sim_result.py。"
        ) from exc
    time_sec = load_series(mat_data, "tout")
    total_hours = int(np.floor(np.nanmax(time_sec) / 3600.0))
    hours = np.arange(total_hours)
    days = hours / 24.0
    demand_kw = load_demand_kw(boundary_file, total_hours)

    data = {
        "hour": hours,
        "day": days,
    }

    station_supply, _ = load_optional_series(mat_data, [
        "T_station_supply", "T_station_supply_K", "station_supply_temp", "T_supply_main"
    ])
    station_return, _ = load_optional_series(mat_data, [
        "T_station_return", "T_station_return_K", "station_return_temp", "T_return_main"
    ])
    station_cooling, station_cooling_name = load_optional_series(mat_data, [
        "Q_station_cooling", "Q_cooling_removed", "station_cooling_W", "Q_cool"
    ])
    if station_supply is not None:
        data["T_station_supply_C"] = hourly_average(
            maybe_kelvin_to_celsius(station_supply), time_sec, total_hours
        )
    if station_return is not None:
        data["T_station_return_C"] = hourly_average(
            maybe_kelvin_to_celsius(station_return), time_sec, total_hours
        )
    if station_cooling is not None:
        q = hourly_average(station_cooling, time_sec, total_hours)
        if np.nanmax(np.abs(q)) > 100000:
            q = q / 1000.0
        data[f"{station_cooling_name}_kw"] = q

    for user in USERS:
        flow = np.abs(load_series(mat_data, f"real_flow_{user}"))
        t_sup = maybe_kelvin_to_celsius(load_series(mat_data, f"T_sup_{user}"))
        t_ret = maybe_kelvin_to_celsius(load_series(mat_data, f"T_ret_{user}"))

        flow_h = hourly_average(flow, time_sec, total_hours)
        t_sup_h = hourly_average(t_sup, time_sec, total_hours)
        t_ret_h = hourly_average(t_ret, time_sec, total_hours)
        delta_t_h = t_ret_h - t_sup_h
        q_sup_kw = np.maximum(flow_h * CP_WATER_KJ_PER_KG_K * delta_t_h, 0.0)

        data[f"user{user}_flow_kg_s"] = flow_h
        data[f"user{user}_T_sup_C"] = t_sup_h
        data[f"user{user}_T_ret_C"] = t_ret_h
        data[f"user{user}_delta_T_C"] = delta_t_h
        data[f"user{user}_Q_sup_kw"] = q_sup_kw

        if user in demand_kw:
            dem = np.asarray(demand_kw[user], dtype=float)[:total_hours]
            data[f"user{user}_demand_kw"] = dem
            data[f"user{user}_Q_unmet_kw"] = np.maximum(dem - q_sup_kw, 0.0)

    df = pd.DataFrame(data)
    q_cols = [f"user{u}_Q_sup_kw" for u in USERS if f"user{u}_Q_sup_kw" in df]
    if q_cols:
        df["total_Q_sup_kw"] = df[q_cols].sum(axis=1)
    demand_cols = [f"user{u}_demand_kw" for u in USERS if f"user{u}_demand_kw" in df]
    if demand_cols:
        df["total_demand_kw"] = df[demand_cols].sum(axis=1)
        df["total_Q_unmet_kw"] = np.maximum(df["total_demand_kw"] - df["total_Q_sup_kw"], 0.0)
    return df


def save_figure(fig, output_dir, name):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_flows(df, output_dir):
    fig, ax = plt.subplots(figsize=(13, 5))
    for user in USERS:
        ax.plot(df["day"], df[f"user{user}_flow_kg_s"], label=f"用户{user}")
    ax.set_title("各用户实际流量")
    ax.set_xlabel("仿真时间 / 天")
    ax.set_ylabel("质量流量 / kg/s")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3)
    return save_figure(fig, output_dir, "01_各用户实际流量.png")


def plot_temperatures(df, output_dir):
    fig, axes = plt.subplots(len(USERS), 1, figsize=(13, 9), sharex=True)
    for ax, user in zip(axes, USERS):
        ax.plot(df["day"], df[f"user{user}_T_sup_C"], label="供水温度", color="#1f77b4")
        ax.plot(df["day"], df[f"user{user}_T_ret_C"], label="回水温度", color="#d62728")
        ax.set_title(f"用户{user}供回水温度")
        ax.set_ylabel("温度 / ℃")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left")
    axes[-1].set_xlabel("仿真时间 / 天")
    return save_figure(fig, output_dir, "02_各用户供回水温度.png")


def plot_station_temperature(df, output_dir):
    station_cols = [c for c in ["T_station_supply_C", "T_station_return_C"] if c in df]
    if not station_cols:
        return None

    fig, ax = plt.subplots(figsize=(13, 5))
    if "T_station_supply_C" in df:
        ax.plot(df["day"], df["T_station_supply_C"], label="冷站出水温度", color="#1f77b4")
    if "T_station_return_C" in df:
        ax.plot(df["day"], df["T_station_return_C"], label="冷站回水温度", color="#d62728")
    ax.axhline(SUPPLY_TEMP_SET_C, color="black", linestyle="--", linewidth=1, alpha=0.65, label="供水设定 7℃")
    ax.axhline(SUPPLY_TEMP_WARN_C, color="#1f77b4", linestyle=":", linewidth=1, alpha=0.75, label="供水报警 12℃")
    ax.axhline(RETURN_TEMP_MAX_C, color="#d62728", linestyle=":", linewidth=1, alpha=0.75, label="回水上限 20℃")
    ax.set_title("冷站供回水温度")
    ax.set_xlabel("仿真时间 / 天")
    ax.set_ylabel("温度 / ℃")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return save_figure(fig, output_dir, "06_冷站供回水温度.png")


def plot_station_cooling(df, output_dir):
    q_cols = [c for c in df.columns if c.startswith("Q_station") or c.startswith("Q_cooling") or c.startswith("station_cooling")]
    if not q_cols:
        return None

    fig, ax = plt.subplots(figsize=(13, 5))
    for col in q_cols:
        ax.plot(df["day"], df[col], label=col)
    if "total_demand_kw" in df:
        ax.plot(df["day"], df["total_demand_kw"], label="总需求", color="black", linestyle="--", alpha=0.75)
    ax.set_title("冷站制冷量")
    ax.set_xlabel("仿真时间 / 天")
    ax.set_ylabel("kW")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return save_figure(fig, output_dir, "07_冷站制冷量.png")


def plot_delta_t(df, output_dir):
    fig, ax = plt.subplots(figsize=(13, 5))
    for user in USERS:
        ax.plot(df["day"], df[f"user{user}_delta_T_C"], label=f"用户{user}")
    ax.axhline(5.0, color="black", linestyle="--", linewidth=1, alpha=0.6, label="设计ΔT 5℃")
    ax.set_title("各用户供回水温差")
    ax.set_xlabel("仿真时间 / 天")
    ax.set_ylabel("ΔT / ℃")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=4)
    return save_figure(fig, output_dir, "03_各用户供回水温差.png")


def plot_cooling(df, output_dir):
    fig, ax = plt.subplots(figsize=(13, 5))
    for user in USERS:
        ax.plot(df["day"], df[f"user{user}_Q_sup_kw"], label=f"用户{user}实际供冷")
        demand_col = f"user{user}_demand_kw"
        if demand_col in df:
            ax.plot(df["day"], df[demand_col], linestyle="--", linewidth=1.1, alpha=0.8, label=f"用户{user}需求")
    ax.set_title("各用户需求与实际供冷量")
    ax.set_xlabel("仿真时间 / 天")
    ax.set_ylabel("冷量 / kW")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3)
    return save_figure(fig, output_dir, "04_各用户需求与实际供冷.png")


def plot_unmet(df, output_dir):
    unmet_cols = [f"user{u}_Q_unmet_kw" for u in USERS if f"user{u}_Q_unmet_kw" in df]
    if not unmet_cols:
        return None
    fig, ax = plt.subplots(figsize=(13, 5))
    for user in USERS:
        col = f"user{user}_Q_unmet_kw"
        if col in df:
            ax.plot(df["day"], df[col], label=f"用户{user}")
    ax.set_title("各用户未满足冷量")
    ax.set_xlabel("仿真时间 / 天")
    ax.set_ylabel("未满足冷量 / kW")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3)
    return save_figure(fig, output_dir, "05_各用户未满足冷量.png")


def plot_dashboard(df, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    ax = axes[0, 0]
    for user in USERS:
        ax.plot(df["day"], df[f"user{user}_flow_kg_s"], label=f"用户{user}")
    ax.set_title("实际流量")
    ax.set_ylabel("kg/s")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3)

    ax = axes[0, 1]
    for user in USERS:
        ax.plot(df["day"], df[f"user{user}_delta_T_C"], label=f"用户{user}")
    ax.axhline(5.0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title("供回水温差")
    ax.set_ylabel("℃")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    if "total_demand_kw" in df:
        ax.plot(df["day"], df["total_demand_kw"], label="总需求", color="black", linestyle="--")
    ax.plot(df["day"], df["total_Q_sup_kw"], label="总实际供冷", color="#2ca02c")
    ax.set_title("总站需求与实际供冷")
    ax.set_xlabel("仿真时间 / 天")
    ax.set_ylabel("kW")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    if "total_Q_unmet_kw" in df:
        ax.plot(df["day"], df["total_Q_unmet_kw"], label="总未满足冷量", color="#d62728")
    else:
        for user in USERS:
            ax.plot(df["day"], df[f"user{user}_Q_sup_kw"], label=f"用户{user}")
    ax.set_title("未满足冷量")
    ax.set_xlabel("仿真时间 / 天")
    ax.set_ylabel("kW")
    ax.grid(True, alpha=0.25)
    ax.legend()

    return save_figure(fig, output_dir, "00_仿真结果总览.png")


def write_summary(df, output_dir):
    rows = []
    for user in USERS:
        rows.append({
            "user_id": user,
            "avg_flow_kg_s": df[f"user{user}_flow_kg_s"].mean(),
            "max_flow_kg_s": df[f"user{user}_flow_kg_s"].max(),
            "avg_T_sup_C": df[f"user{user}_T_sup_C"].mean(),
            "avg_T_ret_C": df[f"user{user}_T_ret_C"].mean(),
            "avg_delta_T_C": df[f"user{user}_delta_T_C"].mean(),
            "avg_Q_sup_kw": df[f"user{user}_Q_sup_kw"].mean(),
            "max_Q_sup_kw": df[f"user{user}_Q_sup_kw"].max(),
            "avg_Q_unmet_kw": df[f"user{user}_Q_unmet_kw"].mean()
            if f"user{user}_Q_unmet_kw" in df else np.nan,
            "max_Q_unmet_kw": df[f"user{user}_Q_unmet_kw"].max()
            if f"user{user}_Q_unmet_kw" in df else np.nan,
        })
    summary = pd.DataFrame(rows)
    summary_path = output_dir / "仿真结果统计摘要.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    station_summary_path = None
    station_cols = [c for c in ["T_station_supply_C", "T_station_return_C"] if c in df]
    if station_cols:
        rows = {}
        for col in station_cols:
            rows[f"{col}_mean"] = df[col].mean()
            rows[f"{col}_min"] = df[col].min()
            rows[f"{col}_max"] = df[col].max()
        station_summary = pd.DataFrame([rows])
        station_summary_path = output_dir / "冷站温度统计摘要.csv"
        station_summary.to_csv(station_summary_path, index=False, encoding="utf-8-sig")
    return summary_path


def main():
    parser = argparse.ArgumentParser(description="读取 Simulink 仿真结果并输出流量、供回水温度等图表。")
    parser.add_argument("--mat", default="sim_result.mat", help="Simulink 结果 MAT 文件")
    parser.add_argument("--boundary", default="Simulink_30Days_UserBoundary.csv", help="用户需求边界 CSV")
    parser.add_argument("--out", default="sim_result_plots", help="图表输出目录")
    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = build_hourly_dataframe(args.mat, args.boundary)
    hourly_path = output_dir / "仿真结果_逐时数据.csv"
    df.to_csv(hourly_path, index=False, encoding="utf-8-sig")

    paths = [
        plot_dashboard(df, output_dir),
        plot_flows(df, output_dir),
        plot_temperatures(df, output_dir),
        plot_station_temperature(df, output_dir),
        plot_delta_t(df, output_dir),
        plot_cooling(df, output_dir),
        plot_station_cooling(df, output_dir),
        plot_unmet(df, output_dir),
        hourly_path,
        write_summary(df, output_dir),
    ]

    if "T_station_supply_C" in df:
        supply_mean = df["T_station_supply_C"].mean()
        supply_max = df["T_station_supply_C"].max()
        if supply_max > SUPPLY_TEMP_WARN_C:
            print(
                f"警告: 冷站出水温度最高 {supply_max:.2f} ℃，超过 {SUPPLY_TEMP_WARN_C:.1f} ℃，"
                "请检查冷站换热器是否串在主水路、冷源方向和制冷量上限。"
            )
        else:
            print(f"冷站出水温度均值 {supply_mean:.2f} ℃，最高 {supply_max:.2f} ℃。")
    if "T_station_return_C" in df:
        return_max = df["T_station_return_C"].max()
        if return_max > RETURN_TEMP_MAX_C:
            print(
                f"警告: 冷站回水温度最高 {return_max:.2f} ℃，超过 {RETURN_TEMP_MAX_C:.1f} ℃，"
                "请检查冷站制冷量、用户热源限幅和水量分配。"
            )
        else:
            print(f"冷站回水温度最高 {return_max:.2f} ℃，未超过 {RETURN_TEMP_MAX_C:.1f} ℃。")

    print("已输出以下文件:")
    for path in paths:
        if path is not None:
            print(f"  {path}")


if __name__ == "__main__":
    main()
