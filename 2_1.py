#获取身高体重
height=float(input("请输入身高(米)"));
weight=float(input("请输入体重（千克）"));
#计算BMI
bmi=weight/height**2
#输出结果
print("你的bmi指数为",bmi,"，健康等级为：");
if bmi<18.5:
    print("偏瘦");
elif bmi<24:
    print(" 正常");
elif bmi<28:        
    print("偏胖");          
else:
    print("肥胖");  