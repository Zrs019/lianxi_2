import pandas as pd
import re
import os

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
    
    for i, row in df.iterrows():
        # 跳过标记的行
        if skip_next_n_rows > 0:
            skip_next_n_rows -= 1
            continue
        
        # 获取非空值
        non_null_values = [str(x).strip() for x in row.values if pd.notna(x) and str(x).strip() != '']
        
        # 跳过空行
        if not non_null_values:
            continue
            
        # 检查是否是各种标题行
        row_text = ' '.join(non_null_values).lower()
        
        # 1. 跳过"截止时间"行
        if '截止时间' in row_text:
            print(f"跳过'截止时间'行: 第{i+1}行")
            continue
            
        # 2. 跳过列头行（包含"设备编号"、"采集时间"、"冷量"等关键词）
        if '设备编号' in row_text or ('设备' in row_text and '时间' in row_text):
            print(f"跳过列头行: 第{i+1}行 - {row_text}")
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
        # 数据行的特征：第一列应该是有效的设备标识（可能是十六进制字符串）
        # 先检查是否有足够的数据列
        if len(non_null_values) >= 3:
            device_id = non_null_values[0]
            time_str = non_null_values[1]
            cooling_str = non_null_values[2]
            
            # 清理时间格式
            time_str = re.sub(r'[：:]', ':', time_str)
            
            # 检查设备ID是否看起来像有效的设备标识（十六进制字符串或数字）
            # 设备ID通常是类似e04b410050ad00005f15010000000000这样的十六进制字符串
            if (len(device_id) >= 10 and 
                (device_id.startswith('e04b') or  # 根据你的数据特点
                 re.match(r'^[a-fA-F0-9]+$', device_id) or  # 十六进制
                 re.match(r'^\d+$', device_id))):  # 纯数字
                
                # 尝试转换冷量为数值
                try:
                    # 处理冷量可能包含的单位或空格
                    cooling_str_clean = re.sub(r'[^\d\.\-]', '', cooling_str)
                    cooling = float(cooling_str_clean) if cooling_str_clean else 0.0
                    
                    data_rows.append([device_id, time_str, cooling])
                    print(f"添加数据行: 第{i+1}行 - 设备{device_id[:8]}..., 时间{time_str}, 冷量{cooling}")
                    
                except ValueError:
                    print(f"冷量转换失败，跳过第{i+1}行: 冷量值='{cooling_str}'")
                    
            else:
                # 第一列不是有效的设备标识
                if device_id not in ['标题', '设备编号', '采集时间', '冷量', '数据类型']:
                    print(f"跳过非数据行: 第{i+1}行 - 首列'{device_id}'不是有效设备ID")
                continue
        else:
            # 列数不足
            print(f"跳过列数不足的行: 第{i+1}行 - {non_null_values}")
            continue
    
    # 创建DataFrame
    if not data_rows:
        print("未找到有效数据行")
        return pd.DataFrame()
    
    cleaned_df = pd.DataFrame(data_rows, columns=['设备编号', '采集时间', '冷量'])
    
    print(f"\n清理后数据形状: {cleaned_df.shape}")
    print("清理后的数据前5行:")
    print(cleaned_df.head())
    
    # 按时间分组求和
    result_df = cleaned_df.groupby('采集时间')['冷量'].sum().reset_index()
    result_df.columns = ['时间', '总冷量']
    result_df = result_df.sort_values('总冷量', ascending=False)
    
    print(f"\n分组汇总结果形状: {result_df.shape}")
    
    return result_df

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
    for _, row in result_df.iterrows():
        print(f"{row['时间']} {int(row['总冷量'])}")
    
    # 保存结果 - 使用绝对路径确保文件位置明确
    output_filename = '3站10月_总冷量_新格式.csv'
    
    # 获取当前工作目录
    current_dir = os.getcwd()
    full_output_path = os.path.join(current_dir, output_filename)
    
    print(f"\n正在保存文件到: {full_output_path}")
    
    try:
        result_df.to_csv(full_output_path, index=False, encoding='utf-8-sig')
        print(f"✅ 结果已保存到 '{output_filename}'")
        
        # 检查文件是否真的存在
        if os.path.exists(full_output_path):
            file_size = os.path.getsize(full_output_path)
            print(f"✅ 文件确认存在，大小: {file_size} 字节")
        else:
            print(f"❌ 警告: 文件保存后未找到: {full_output_path}")
            
    except Exception as e:
        print(f"❌ 保存文件时出错: {e}")
    
    print("\n" + "="*50)
    print("🎯 当前目录文件列表")
    print("="*50)
    
    # 列出当前目录所有文件
    print(f"当前工作目录: {current_dir}")
    print("\n当前目录下的所有文件:")
    files_in_dir = os.listdir(current_dir)
    csv_files = [f for f in files_in_dir if f.endswith('.csv')]
    
    if csv_files:
        print("📄 CSV文件:")
        for csv_file in csv_files:
            full_path = os.path.join(current_dir, csv_file)
            file_size = os.path.getsize(full_path)
            print(f"  - {csv_file} ({file_size} 字节)")
    else:
        print("⚠️ 当前目录没有CSV文件")
    
    print("\n📁 其他文件:")
    for file in files_in_dir:
        if not file.endswith('.csv'):
            print(f"  - {file}")
    
    # 提供可直接点击的链接
    if os.path.exists(full_output_path):
        print(f"\n🔗 文件链接: file://{full_output_path}")
else:
    print("处理失败：未提取到有效数据")