import argparse
from pathlib import Path

import pandas as pd


def parse_col_spec(spec: str):
    """Return int index if spec is numeric, else return column name."""
    try:
        return int(spec)
    except ValueError:
        return spec


def get_series(df: pd.DataFrame, col_spec):
    """Get a column by index or name."""
    if isinstance(col_spec, int):
        if col_spec < 0 or col_spec >= len(df.columns):
            raise IndexError(f"Column index {col_spec} is out of range.")
        return df.iloc[:, col_spec]
    if col_spec not in df.columns:
        raise KeyError(f"Column '{col_spec}' not found. Available columns: {list(df.columns)}")
    return df[col_spec]


def load_table(file_path: Path, sheet_name=0, has_header=True) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    header = 0 if has_header else None

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path, sheet_name=sheet_name, header=header)
    if suffix == ".csv":
        return pd.read_csv(file_path, header=header)

    raise ValueError("Only .csv, .xlsx, .xls are supported.")


def extract_midnight_rows(df: pd.DataFrame, time_col, value_col) -> pd.DataFrame:
    time_raw = get_series(df, time_col)
    value_raw = get_series(df, value_col)

    parsed_time = pd.to_datetime(time_raw, errors="coerce")
    mask = (
        parsed_time.notna()
        & (parsed_time.dt.hour == 0)
        & (parsed_time.dt.minute == 0)
        & (parsed_time.dt.second == 0)
    )

    result = pd.DataFrame(
        {
            "time": parsed_time[mask],
            "value": value_raw[mask],
        }
    ).sort_values("time")

    return result.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Extract rows where time is exactly 00:00:00 from a two-column table."
    )
    parser.add_argument("input", help="Input file path (.csv/.xlsx/.xls)")
    parser.add_argument("-o", "--output", help="Output csv path")
    parser.add_argument(
        "--time-col",
        default="0",
        help="Time column index/name. Default: 0 (first column).",
    )
    parser.add_argument(
        "--value-col",
        default="1",
        help="Value column index/name. Default: 1 (second column).",
    )
    parser.add_argument(
        "--sheet",
        default="0",
        help="Excel sheet name/index. Default: 0",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Use this if your file has no header row.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(f"{input_path.stem}_midnight.csv")
    )

    time_col = parse_col_spec(args.time_col)
    value_col = parse_col_spec(args.value_col)

    sheet_spec = parse_col_spec(args.sheet)
    df = load_table(input_path, sheet_name=sheet_spec, has_header=not args.no_header)
    result = extract_midnight_rows(df, time_col=time_col, value_col=value_col)

    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Input rows: {len(df)}")
    print(f"Midnight rows: {len(result)}")
    print(f"Output saved to: {output_path}")

    if not result.empty:
        print("\nPreview:")
        print(result.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
