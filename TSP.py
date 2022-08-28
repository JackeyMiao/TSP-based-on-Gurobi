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
rnd.seed(31415926) # ������ӣ����б��У������ÿ�����н��һ���������⸳ֵ

n = 100 # һ����������
xc = rnd.rand(n)*100 # �������ÿ�����еĺ����꣬��Χ[0,100]
yc = rnd.rand(n)*100 # �������ÿ�����е������꣬��Χ[0,100]

# ���Ի�ͼ��һ�����ɵĳ���ʲô����
# plt.plot(xc[0], yc[0], c='r',marker='s' ) # ����Ϊ0�ĵ㣬��depot/�ֿ�/������
# plt.scatter(xc, yc, c='b') # �ͻ���
# plt.show()

V = list(range(0,n)) # ���е㼯��
A = [(i,j) for i in V for j in V if i !=j] # ����֮������Щ·
C = {(i,j): np.hypot(xc[i]-xc[j], yc[i]-yc[j]) for i,j in A} # np.hypot��������=��ƽ���ͣ����㻡�εĳ���


mdl = Model('TSP') # ������

x = mdl.addVars(A, vtype=GRB.BINARY) # ���ӱ���xij����ʾ�Ƿ�����ij
y = mdl.addVars(A, vtype=GRB.INTEGER) # ���ӱ���ui����ʾ���ڸõ㴦�ۼ��ػ���

mdl.modelSense = GRB.MINIMIZE # Ŀ��Ϊ��С��
mdl.setObjective(quicksum(x[i,j]*C[i,j] for i,j in A )) # Ŀ�꺯��Ϊ�ܾ���

# �������Լ��(��ģ��ʽ��Gavish-Graves Formulation)
mdl.addConstrs(quicksum(x[i,j] for j in V if i != j)==1 for i in V) 
mdl.addConstrs(quicksum(x[i,j] for i in V if i!=j)==1 for j in V)
mdl.addConstrs((quicksum(y[i,j] for j in V if i!=j) - quicksum(y[j,i] for j in V if i!=j and j!=0) == 1) for i in V if i!=0)
mdl.addConstrs(y[i,j] <= (n-1)*x[i,j] for i in V if i!=0 for j in V if i!=j)
mdl.addConstrs(y[i,j] >= 0 for i,j in A)

mdl.optimize() # �Ż�

#�Ż���ɣ�����������
active_arts = [a for a in A if x[a].x > 0.9] # ������Ž���������ߣ���xij����1��(i,j)
# ���ڴ�����xij����Ϊ0.999999999����˲�Ҫ��==1
print(active_arts)

# ��ͼ
for index, (i,j) in enumerate(active_arts):
    plt.plot([xc[i],xc[j]],[yc[i],yc[j]],c='r')
plt.scatter(xc, yc, c='b')
plt.title('TSP'+str(n))
plt.show()