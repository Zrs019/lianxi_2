items=list(map(int, input("输入整数，用逗号分隔").split(",")));
sum=0
for i in items:
    if i%2==0:
        sum=sum+i**2;
print("偶数的平方和为：",sum);