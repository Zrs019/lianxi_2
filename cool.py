import pandas as pd
import re
import os
from datetime import datetime


# 如果有总表/不参与统计的设备编号，在这里维护
EXCLUDED_DEVICE_IDS = {
    '864601065981953',
}


def _norm_text(value):
    """标准化单元格文本，便于做关键词匹配。"""
    if pd.isna(value):
        return ""
    return str(value).strip()


def _find_header_indexes(row_values):
    """在一行中定位设备编号/采集时间/冷量所在列索引。"""
    idx_map = {'设备编号': None, '采集时间': None, '冷量': None}
    for idx, val in enumerate(row_values):
        text = _norm_text(val)
        if not text:
            continue
        if idx_map['设备编号'] is None and ('设备编号' in text or text == '设备'):
            idx_map['设备编号'] = idx
        if idx_map['采集时间'] is None and ('采集时间' in text or '时间' == text):
            idx_map['采集时间'] = idx
        if idx_map['冷量'] is None and '冷量' in text:
            idx_map['冷量'] = idx

    if all(v is not None for v in idx_map.values()):
        return idx_map
    return None


def _parse_cooling_value(cooling_str):
    """把冷量字段安全转换成 float。"""
    if pd.isna(cooling_str):
        return None
    text = _norm_text(cooling_str)
    if not text:
        return None
    # 保留数字、符号、小数点；去掉单位、逗号等
    cleaned = re.sub(r'[^\d\.\-]', '', text)
    if cleaned in ('', '-', '.', '-.'):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def process_cooling_data_new_format(file_path):
    """
    处理新格式的冷量数据
    标题行包含：截止时间、数据类型、表格列头
    
    Parameters:
    file_path: Excel文件路径
    """
    # 读取Excel，不自动识别标题
    df = pd.read_excel(file_path, header=None, dtype=str)
    
    print(f"处理文件: {file_path}")
    print("原始数据预览:")
    print(df.head(10))
    print(f"\n原始数据形状: {df.shape}")
    
    # 存储清理后的数据
    data_rows = []
    skip_next_n_rows = 0  # 用于跳过特定行数
    header_idx = None
    
    for i, row in df.iterrows():
        # 跳过标记的行
        if skip_next_n_rows > 0:
            skip_next_n_rows -= 1
            continue
        
        # 获取文本形式的行内容
        row_values = list(row.values)
        non_null_values = [_norm_text(x) for x in row_values if _norm_text(x) != '']
        
        # 跳过空行
        if not non_null_values:
            continue
            
        # 检查是否是各种标题行
        row_text = ' '.join(non_null_values).lower()
        
        # 1. 跳过"截止时间"行
        if '截止时间' in row_text:
            print(f"跳过'截止时间'行: 第{i+1}行")
            continue
            
        # 2. 识别列头行（记录列位置，后续按列读取，避免错列）
        detected_header = _find_header_indexes(row_values)
        if detected_header:
            header_idx = detected_header
            print(f"识别到列头: 第{i+1}行 - {header_idx}")
            # 如果下一行可能是"瞬时值"等描述行，也跳过
            if i + 1 < len(df):
                next_row_text = ' '.join([str(x).strip() for x in df.iloc[i+1].values if pd.notna(x) and str(x).strip() != ''])
                if '瞬时值' in next_row_text or '累计值' in next_row_text:
                    skip_next_n_rows = 1
                    print(f"  同时跳过下一行: '瞬时值'或'累计值'描述行")
            continue
            
        # 3. 跳过"瞬时值"、"累计值"等描述行
        if '瞬时值' in row_text or '累计值' in row_text:
            print(f"跳过描述行: 第{i+1}行")
            continue
            
        # 4. 检查是否是数据行
        # 若已识别列头，则严格按列索引读取；避免“前三个非空值”错位
        if header_idx is None:
            print(f"跳过未识别列头前的数据行: 第{i+1}行")
            continue

        device_id = _norm_text(row_values[header_idx['设备编号']])
        time_str = _norm_text(row_values[header_idx['采集时间']])
        cooling_str = _norm_text(row_values[header_idx['冷量']])

        if not (device_id and time_str and cooling_str):
            print(f"跳过关键字段缺失行: 第{i+1}行")
            continue

        # 清理时间格式
        time_str = re.sub(r'[：]', ':', time_str)
        # 统一时间精度，避免同一分钟因格式差异分成多组
        parsed_time = pd.to_datetime(time_str, errors='coerce')
        if pd.isna(parsed_time):
            print(f"时间解析失败，跳过第{i+1}行: 时间值='{time_str}'")
            continue
        time_str = parsed_time.strftime('%Y-%m-%d %H:%M:%S')

        # 检查设备ID是否看起来像有效设备标识
        if not (
            len(device_id) >= 10
            and (
                device_id.startswith('e04b')
                or re.match(r'^[a-fA-F0-9]+$', device_id)
                or re.match(r'^\d+$', device_id)
            )
        ):
            if device_id not in ['标题', '设备编号', '采集时间', '冷量', '数据类型']:
                print(f"跳过非数据行: 第{i+1}行 - 设备ID无效 '{device_id}'")
            continue

        cooling = _parse_cooling_value(cooling_str)
        if cooling is None:
            print(f"冷量转换失败，跳过第{i+1}行: 冷量值='{cooling_str}'")
            continue

        data_rows.append([device_id, time_str, cooling])
        print(f"添加数据行: 第{i+1}行 - 设备{device_id[:8]}..., 时间{time_str}, 冷量{cooling}")
    
    # 创建DataFrame
    if not data_rows:
        print("未找到有效数据行")
        return pd.DataFrame()
    
    cleaned_df = pd.DataFrame(data_rows, columns=['设备编号', '采集时间', '冷量'])

    # 按配置排除不参与统计的设备（例如总表设备）
    excluded_rows = cleaned_df[cleaned_df['设备编号'].isin(EXCLUDED_DEVICE_IDS)].copy()
    if not excluded_rows.empty:
        print("\n检测到并排除以下设备编号:")
        print(sorted(excluded_rows['设备编号'].unique().tolist()))
        by_time_excluded = excluded_rows.groupby('采集时间', as_index=False)['冷量'].sum()
        by_time_excluded.columns = ['时间', '被排除冷量']
        print("被排除设备在各时刻的冷量贡献:")
        print(by_time_excluded.sort_values('时间').reset_index(drop=True))

    cleaned_df = cleaned_df[~cleaned_df['设备编号'].isin(EXCLUDED_DEVICE_IDS)].copy()
    
    print(f"\n清理后数据形状(排除后): {cleaned_df.shape}")
    print("清理后的数据前5行:")
    print(cleaned_df.head())
    
    # 口径1：按时间分组求和（同一时刻各设备读数直接相加）
    result_df = cleaned_df.groupby('采集时间', as_index=False)['冷量'].sum()
    result_df.columns = ['时间', '总冷量']
    # 统一保留两位小数，避免展示时因浮点误差造成对账困扰
    result_df['总冷量'] = result_df['总冷量'].round(2)

    # 两种排序都提供：峰值排序 + 时间排序
    result_by_peak = result_df.sort_values('总冷量', ascending=False).reset_index(drop=True)
    result_by_time = result_df.sort_values('时间', ascending=True).reset_index(drop=True)
    
    print(f"\n分组汇总结果形状: {result_df.shape}")
    print("\n按峰值降序:")
    print(result_by_peak)
    print("\n按时间升序(便于手工对账):")
    print(result_by_time)

    # 口径2：按设备做相邻时间差分，再按时间汇总（常用于累计表计）
    diff_df = cleaned_df.copy()
    diff_df['采集时间_dt'] = pd.to_datetime(diff_df['采集时间'])
    diff_df = diff_df.sort_values(['设备编号', '采集时间_dt'])
    diff_df['冷量增量'] = diff_df.groupby('设备编号')['冷量'].diff()
    # 首次出现设备记为 0；若出现回表/重置导致负增量，按当前值计入
    diff_df['冷量增量'] = diff_df['冷量增量'].fillna(0)
    neg_mask = diff_df['冷量增量'] < 0
    diff_df.loc[neg_mask, '冷量增量'] = diff_df.loc[neg_mask, '冷量']

    result_delta = diff_df.groupby('采集时间', as_index=False)['冷量增量'].sum()
    result_delta.columns = ['时间', '总冷量增量']
    result_delta['总冷量增量'] = result_delta['总冷量增量'].round(2)
    result_delta = result_delta.sort_values('时间').reset_index(drop=True)

    print("\n按设备差分后的时间增量(用于累计值口径对账):")
    print(result_delta)
    
    return result_by_peak

# 使用函数处理数据
result_df = process_cooling_data_new_format(r"D:\study\和达能源站\zrs2026\峰值处理\原始\3站十月数据.xls")  # 修改为你的测试文件

if not result_df.empty:
    print("\n" + "="*50)
    print("每个时刻的总冷量:")
    print("="*50)
    print(result_df)
    
    print("\n" + "="*50)
    print("时间-总冷量格式:")
    print("="*50)
    for _, row in result_df.sort_values('时间').iterrows():
        print(f"{row['时间']} {row['总冷量']:.2f}")
    
    # 保存结果
    output_csv = r"D:\study\和达能源站\zrs2026\峰值处理\原始\3站十月数据_总冷量_新格式.csv"
    try:
        result_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到: {output_csv}")
    except PermissionError:
        fallback_csv = output_csv.replace(
            '.csv',
            f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        result_df.to_csv(fallback_csv, index=False, encoding='utf-8-sig')
        output_csv = fallback_csv
        print(f"\n原文件被占用，已另存为: {output_csv}")
    
    print("\n" + "="*50)
    print("🎯 生成文件地址信息")
    print("="*50)
    
    # 直接检查实际输出文件路径
    if os.path.exists(output_csv):
        print(f"📂 文件保存在：{output_csv}")
    else:
        print(f"⚠️ 输出文件未生成：{output_csv}")
else:
    print("处理失败：未提取到有效数据")