import numpy as np

scores = np.array([
    [85, 90, 78, 92],   # 学生1
    [76, 88, 95, 81],   # 学生2
    [92, 75, 88, 96],   # 学生3
    [68, 82, 73, 79],   # 学生4
    [95, 91, 89, 87],   # 学生5
])

# 1. 求每个学生的平均分（输出5个数）
print("每个学生的平均分：", np.mean(scores, axis=1))
# 2. 求每门课的最高分（输出4个数）
print("每门课的最高分：",np.max(scores,axis=0) )
# 3. 找出总分最高的学生是第几个（输出索引）
print("总分最高的学生是第几个：",np.argmax(np.sum(scores,axis=1)))
# 4. 将所有 < 80 的成绩标记为 "不及格"，>= 80 标记为 "及格"（用 np.where）
print("成绩标记结果：",np.where(scores >= 80, "及格", "不及格"))
# 5. 对成绩做标准化（每门课：(x - 均值) / 标准差），输出标准化后的矩阵
b=np.std(scores,axis=0)
print("标准化后的成绩：",(scores - np.mean(scores,axis=0)) / np.std(scores,axis=0))