import pandas as pd

# 1. 加载 Excel 文件
file_path = r"duqudaochu/output/3号能源站-A线+3号能源站-B线+3号能源站-C线+3号能源站-D线+中心能源站-维亚园区+中心能源站-中心站地块+中心能源站-加速器五期高区+中心能源站-加速器五期低区+中心能源站-康洲园区_2025-08-01_00-00-00_2025-09-01_00-00-00_冷量汇总.xlsx"  # 请替换为你的实际文件名
df = pd.read_excel(file_path)

# 2. 定义需要合并的目标线路
target_lines = ['A线', 'B线', 'C线', 'D线', '四期']

# 3. 核心处理流程：
#    - 假设第一列是线路名称，第二列是时间，第三列是冷量
#    - 如果你的列名不是这些，代码会根据索引位置（0, 1, 2）进行处理
name_col = df.columns[0]  # 第一列：线路名
time_col = df.columns[1]  # 第二列：时间
data_col = df.columns[2]  # 第三列：冷量值

# 4. 过滤数据：只保留 A/B/C/D线 和 四期
mask = df[name_col].isin(target_lines)
filtered_df = df[mask].copy()

# 5. 格式化时间（确保时间格式统一，避免因格式微差导致无法合并）
filtered_df[time_col] = pd.to_datetime(filtered_df[time_col])

# 6. 按时间分组并求和
#    这将把同一时间点下的 A、B、C、D、四期的冷量全部加在一起
summary_df = filtered_df.groupby(time_col)[data_col].sum().reset_index()

# 7. 结果输出
print("前5行合并结果示例：")
print(summary_df.head())

# 8. 保存为新的 Excel 文件
summary_df.to_excel("冷量汇总结果.xlsx", index=False)