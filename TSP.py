'''
Description: 
Author: Jackeeee_M
Date: 2022-08-27 13:05:06
LastEditors: Jackeeee_M
LastEditTime: 2022-08-27 16:58:13
'''
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import *

rnd = np.random
rnd.seed(31415926) # 随机种子，如有本行，则程序每次运行结果一样。可任意赋值

n = 100 # 一共几个城市
xc = rnd.rand(n)*100 # 随机生成每个城市的横坐标，范围[0,100]
yc = rnd.rand(n)*100 # 随机生成每个城市的纵坐标，范围[0,100]

# 可以画图看一眼生成的城市什么样子
# plt.plot(xc[0], yc[0], c='r',marker='s' ) # 索引为0的点，即depot/仓库/出发点
# plt.scatter(xc, yc, c='b') # 客户点
# plt.show()

V = list(range(0,n)) # 所有点集合
A = [(i,j) for i in V for j in V if i !=j] # 城市之间有哪些路
C = {(i,j): np.hypot(xc[i]-xc[j], yc[i]-yc[j]) for i,j in A} # np.hypot：二范数=求平方和；计算弧段的长度


mdl = Model('TSP') # 起名字

x = mdl.addVars(A, vtype=GRB.BINARY) # 增加变量xij，表示是否链接ij
y = mdl.addVars(A, vtype=GRB.INTEGER) # 增加变量ui，表示车在该点处累计载货量

mdl.modelSense = GRB.MINIMIZE # 目标为最小化
mdl.setObjective(quicksum(x[i,j]*C[i,j] for i,j in A )) # 目标函数为总距离

# 添加所有约束(建模方式：Gavish-Graves Formulation)
mdl.addConstrs(quicksum(x[i,j] for j in V if i != j)==1 for i in V) 
mdl.addConstrs(quicksum(x[i,j] for i in V if i!=j)==1 for j in V)
mdl.addConstrs((quicksum(y[i,j] for j in V if i!=j) - quicksum(y[j,i] for j in V if i!=j and j!=0) == 1) for i in V if i!=0)
mdl.addConstrs(y[i,j] <= (n-1)*x[i,j] for i in V if i!=0 for j in V if i!=j)
mdl.addConstrs(y[i,j] >= 0 for i,j in A)

mdl.optimize() # 优化

#优化完成，下面输出结果
active_arts = [a for a in A if x[a].x > 0.9] # 输出最优解的所有连线，即xij中是1的(i,j)
# 由于存在误差，xij可能为0.999999999，因此不要用==1
print(active_arts)

# 画图
for index, (i,j) in enumerate(active_arts):
    plt.plot([xc[i],xc[j]],[yc[i],yc[j]],c='r')
plt.scatter(xc, yc, c='b')
plt.title('TSP'+str(n))
plt.show()