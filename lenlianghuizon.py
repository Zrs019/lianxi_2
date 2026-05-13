from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

import pandas as pd


# 1. 在这里填写输入文件地址。也可以填写一个文件夹地址，脚本会处理里面的 Excel/CSV。
INPUT_PATH = r"D:\minicondadaima\lianxi\duqudaochu\output\3号能源站-A线+3号能源站-B线+3号能源站-C线+3号能源站-D线+中心能源站-维亚园区+中心能源站-中心站地块+中心能源站-加速器五期高区+中心能源站-加速器五期低区+中心能源站-康洲园区_2025-08-01_00-00-00_2025-09-01_00-00-00.xlsx"

# 2. 输出文件地址。留空时，自动输出到输入文件旁边，文件名为：原文件名_冷量汇总.xlsx
OUTPUT_PATH = ""


OUTPUT_COLUMNS = ["区域名称", "时间", "冷量汇总"]


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_region(value: object) -> str:
    name = clean_text(value)
    name = re.sub(r"\s+", "", name)
    name = re.sub(r"^(中心能源站[-_—]*)+", "", name)

    if "加速器五期" in name and ("高区" in name or "低区" in name):
        return "加速器五期"
    return name


def parse_number(value: object) -> float | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)

    text = clean_text(value).replace(",", "")
    if not text or text in {"-", "--", "无", "nan", "NaN"}:
        return None

    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if match is None:
        return None
    return float(match.group())


def normalize_time(value: object) -> pd.Timestamp | None:
    if pd.isna(value):
        return None

    if isinstance(value, str):
        value = value.replace("：", ":")

    dt = pd.to_datetime(value, errors="coerce")
    if pd.isna(dt):
        return None
    return pd.Timestamp(dt).replace(tzinfo=None)


def find_col_in_row(
    df: pd.DataFrame,
    row: int,
    includes: tuple[str, ...],
    excludes: tuple[str, ...] = (),
) -> int | None:
    for col in range(df.shape[1]):
        text = clean_text(df.iat[row, col])
        if text and all(word in text for word in includes) and not any(word in text for word in excludes):
            return col
    return None


def find_time_col(df: pd.DataFrame, row: int) -> int | None:
    # 优先使用真正的数据列“采集时间”，避免误识别上方的“开始时间/截止时间”。
    col = find_col_in_row(df, row, ("采集时间",))
    if col is not None:
        return col
    return find_col_in_row(df, row, ("时间",), ("开始", "截止", "结束"))


def find_cooling_instant_col(df: pd.DataFrame, header_row: int) -> int | None:
    max_header_row = min(header_row + 4, len(df))

    # 情况一：单元格本身就是“瞬时冷量”“冷量瞬时值”等。
    for row in range(header_row, max_header_row):
        col = find_col_in_row(df, row, ("冷", "瞬时"), ("热", "累计", "使用", "流量"))
        if col is not None:
            return col

    # 情况二：附件格式，第一行是“冷量（kWh）”，下一行同列是“瞬时值”。
    for parent_col in range(df.shape[1]):
        parent = clean_text(df.iat[header_row, parent_col])
        if "冷" not in parent or "热" in parent:
            continue

        for child_row in range(header_row + 1, max_header_row):
            child = clean_text(df.iat[child_row, parent_col])
            if "瞬时" in child and not any(word in child for word in ("累计", "使用", "热")):
                return parent_col

    return None


def detect_columns(df: pd.DataFrame) -> tuple[int, int, int, int]:
    """
    返回：表头行号、区域列号、时间列号、冷量瞬时值列号。

    附件里的真实表头行为：
    区域名称 | 设备名称 | 设备编号 | 采集时间 | 冷量（kWh）
                                         瞬时值
    """
    for row in range(min(50, len(df))):
        area_col = find_col_in_row(df, row, ("区域",))
        time_col = find_time_col(df, row)
        cooling_col = find_cooling_instant_col(df, row)

        if area_col is not None and time_col is not None and cooling_col is not None:
            return row, area_col, time_col, cooling_col

    raise ValueError("未能识别表头，请确认表格中包含：区域名称、采集时间、冷量（瞬时值）。")


def table_to_records(df: pd.DataFrame) -> list[dict[str, object]]:
    header_row, area_col, time_col, cooling_col = detect_columns(df)
    records: list[dict[str, object]] = []
    last_region = ""

    for row in range(header_row + 1, len(df)):
        row_text = " ".join(clean_text(value) for value in df.iloc[row].tolist())
        if not row_text:
            continue
        if "瞬时值" in row_text or "累计值" in row_text:
            continue
        if "区域" in row_text and "时间" in row_text:
            continue

        raw_region = normalize_region(df.iat[row, area_col])
        region = raw_region or last_region
        time_value = normalize_time(df.iat[row, time_col])
        cooling_value = parse_number(df.iat[row, cooling_col])

        if not region or time_value is None or cooling_value is None:
            continue

        if raw_region:
            last_region = raw_region

        records.append(
            {
                "区域名称": region,
                "时间": time_value,
                "冷量汇总": cooling_value,
            }
        )

    return records


def read_table_file(path: Path) -> list[dict[str, object]]:
    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls", ".xlsm"}:
        records: list[dict[str, object]] = []
        sheets = read_excel_safely(path)
        for df in sheets.values():
            if not df.dropna(how="all").empty:
                records.extend(table_to_records(df))
        return records

    if suffix == ".csv":
        for encoding in ("utf-8-sig", "gbk", "utf-8"):
            try:
                df = pd.read_csv(path, header=None, dtype=object, encoding=encoding)
                return table_to_records(df)
            except UnicodeDecodeError:
                continue

    raise ValueError(f"不支持的文件格式：{path}")


def read_excel_safely(path: Path) -> dict[str, pd.DataFrame]:
    try:
        return pd.read_excel(path, sheet_name=None, header=None, dtype=object)
    except (PermissionError, OSError):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / path.name
            shutil.copy2(path, temp_path)
            return pd.read_excel(temp_path, sheet_name=None, header=None, dtype=object)


def collect_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"输入路径不存在：{input_path}")

    files: list[Path] = []
    for pattern in ("*.xlsx", "*.xls", "*.xlsm", "*.csv"):
        files.extend(input_path.glob(pattern))
    return sorted(path for path in files if not path.name.startswith("~$"))


def process_cooling_files(input_path: str | Path) -> pd.DataFrame:
    paths = collect_files(Path(input_path))
    if not paths:
        raise FileNotFoundError(f"未找到可处理的表格文件：{input_path}")

    records: list[dict[str, object]] = []
    for path in paths:
        records.extend(read_table_file(path))

    if not records:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    raw = pd.DataFrame(records)
    result = (
        raw.groupby(["区域名称", "时间"], as_index=False, sort=True)["冷量汇总"]
        .sum()
        .sort_values(["区域名称", "时间"], kind="stable")
    )
    result["时间"] = result["时间"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return result[OUTPUT_COLUMNS]


def default_output_path(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path.with_name(f"{input_path.stem}_冷量汇总.xlsx")
    return input_path / "冷量汇总.xlsx"


def main() -> None:
    input_path = Path(INPUT_PATH)
    output_path = Path(OUTPUT_PATH) if OUTPUT_PATH else default_output_path(input_path)

    result = process_cooling_files(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".csv":
        result.to_csv(output_path, index=False, encoding="utf-8-sig")
    else:
        result.to_excel(output_path, index=False)

    print(f"处理完成：{len(result)} 行")
    print(f"输出文件：{output_path.resolve()}")


if __name__ == "__main__":
    main()
