class Dataset:
    """
    简易数据集类
    模拟 Pandas DataFrame 的基本功能
    """

    def __init__(self, data, columns):
        """
        data: 二维列表，每行是一条记录
        columns: 列名列表
        """
        self.data = data
        self.columns = columns
        
        self.col_map={};
        for i,col_name in enumerate(self.columns):
            self.col_map[col_name]=i
    
    @property
    def shape(self):
        """返回 (行数, 列数)"""
        rows = len(self.data)
        cols = len(self.columns) if self.data else 0
        return (rows, cols)

    def head(self, n=5):
        """显示前 n 行"""
        print(" | ".join(self.columns))
        print("-" * 40)
        for row in self.data[:n]:
            print(" | ".join(str(item) for item in row))
        print(f"\n[共 {self.shape[0]} 行 × {self.shape[1]} 列]")

    def get_column(self, col_name):
        """获取某一列的数据"""
        if col_name not in self.columns:
            raise KeyError(f"列 '{col_name}' 不存在！")
        idx = self.columns.index(col_name)
        return [row[idx] for row in self.data]

    def describe(self, col_name):
        """对数值列做基本统计"""
        values = self.get_column(col_name)
        nums = [v for v in values if isinstance(v, (int, float))]
        if not nums:
            print(f"列 '{col_name}' 不包含数值数据")
            return
        n = len(nums)
        mean = sum(nums) / n
        sorted_nums = sorted(nums)
        median = sorted_nums[n // 2]
        print(f"📊 列 '{col_name}' 的统计信息：")
        print(f"   数量: {n}")
        print(f"   均值: {mean:.2f}")
        print(f"   中位数: {median}")
        print(f"   最小值: {min(nums)}")
        print(f"   最大值: {max(nums)}")

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return f"Dataset({self.shape[0]} rows × {self.shape[1]} cols)"

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, col_name):
        """支持 dataset['列名'] 的方式访问"""
        return self.get_column(col_name)
    
    def sort_by(self, col_name, ascending=True):
        """按某列排序"""
        col_index = self.col_map[col_name]
        def get_sort_value(row):
            return row[col_index]
        self.data.sort(key=get_sort_value, reverse=not ascending)
        return Dataset(self.data, self.columns)
    
    def show(self, limit=5):
        """显示前limit行"""
        print(' | '.join(self.columns))
        print('-' * 40)
        for i, row in enumerate(self.data):
            if i >= limit:
                print(f"... 共{len(self.data)}行")
                break
            # 把每个单元格转成字符串并用 | 连接
            str_row = [str(cell) for cell in row]
            print(' | '.join(str_row))

    def filter_by(self,col_name, condition):
        col_index = self.col_map[col_name]
        filtered_data=[]
        for row in self.data:
            if condition(row[col_index]):
                filtered_data.append(row.copy())
        return Dataset(filtered_data, self.columns)

    
# ========== 使用 ==========
data = [
    ["张三", 22, 85],
    ["李四", 25, 92],
    ["王五", 20, 78],
    ["赵六", 23, 95],
    ["钱七", 21, 60],
]
columns = ["姓名", "年龄", "成绩"]

ds = Dataset(data, columns)

print(ds)             # Dataset(5 rows × 3 cols)
print(len(ds))        # 5
print(ds.shape)       # (5, 3)

ds.head(3)
# 姓名 | 年龄 | 成绩
# ----------------------------------------
# 张三 | 22 | 85
# 李四 | 25 | 92
# 王五 | 20 | 78
# [共 5 行 × 3 列]

print(ds["成绩"])     # [85, 92, 78, 95, 60]  → __getitem__

ds.describe("成绩")
# 📊 列 '成绩' 的统计信息：
#    数量: 5
#    均值: 82.00
#    中位数: 85
#    最小值: 60
#    最大值: 95
ds.sort_by("年龄", ascending=False).show()
ds.filter_by("成绩", lambda x: x >= 90 ).show()