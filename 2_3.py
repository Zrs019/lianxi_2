import pandas as pd

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Q1: 数据集共有多少行、多少列？
print("hangshu:",df.shape[0])
print("lieshu:",df.shape[1])
# Q2: Age 列有多少个缺失值？缺失率是多少？
missing_age=df["Age"].isnull().sum()
missing_age_rate=missing_age/df.shape[0]
print("Age 列缺失值数量:",missing_age)
print("Age 列缺失率: {:.2%}".format(missing_age_rate))
# Q3: 女性乘客中，存活率是多少？（保留2位小数）
females=df.loc[df["Sex"]=='female']
female_survived=df.loc[(df["Sex"]=="female")&(df["Survived"]==1)]
female_survived_rate=len(female_survived)/len(females)
print("女性乘客存活率: {:.2%}".format(female_survived_rate))
# Q4: 三等舱（Pclass==3）中，年龄最大的乘客叫什么名字？多少岁？
p3=df.loc[df["Pclass"]==3]
P3_top_age=p3["Age"].idxmax()
print("三等舱中，年龄最大的乘客叫:",p3.loc[P3_top_age,"Name"],p3.loc[P3_top_age,"Age"],"岁")

# Q5: 票价（Fare）最贵的前5名乘客的姓名和票价分别是什么？
fare_large=df.nlargest(5,"Fare")
print(fare_large.loc[:,["Name","Fare"]])
#     提示：df.nlargest(5, "Fare") 或 df.sort_values("Fare", ascending=False).head(5)