# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 09:44:27 2024

@author: DELLPHOTO
"""

import gurobipy as gu
import operator
import numpy as np
import itertools

coor = {1:(1,1), 2:(2,2) , 3:(1,3), 4:(3,3)}

BigM = max(coor.values(), key=operator.itemgetter(1))[1]*10000

M = coor.keys()

m = gu.Model('Area')

x_min = m.addVar(vtype=gu.GRB.CONTINUOUS, name="x_min")
x_max = m.addVar(vtype=gu.GRB.CONTINUOUS, name="x_max")
y_min = m.addVar(vtype=gu.GRB.CONTINUOUS, name="y_min")
y_max = m.addVar(vtype=gu.GRB.CONTINUOUS, name="y_max")

A = m.addVar(vtype=gu.GRB.BINARY, name="ayuda")

w = m.addVar(vtype=gu.GRB.CONTINUOUS, name="width")
h = m.addVar(vtype=gu.GRB.CONTINUOUS, name="height")

z = {i:m.addVar(vtype=gu.GRB.BINARY, name="z_"+str(i)) for i in M}

for i in M:
    m.addConstr(x_min <= coor[i][0] + BigM*(1 - z[i]))   
    m.addConstr(coor[i][0] - BigM*(1 - z[i]) <= x_max)
    m.addConstr(y_min <= coor[i][1] + BigM*(1 - z[i]))
    m.addConstr(coor[i][1] - BigM*(1 - z[i]) <= y_max)
    
m.addConstr(x_min <= x_max)
m.addConstr(y_min <= y_max)

m.addConstr(w >= x_max - x_min)
m.addConstr(w <= x_max - x_min)

m.addConstr(h >= y_max - y_min)
m.addConstr(h <= y_max - y_min)

m.addConstr(z[1] == 1)
m.addConstr(z[2] == 1)

m.setObjective(A, sense = gu.GRB.MAXIMIZE)
m.update()
m.setParam('OutputFlag',1)
m.optimize()

print(w.x)
print(h.x)
print(h.x*w.x)
print(m.ObjVal)





'''
for i in triplets:
    m.addConstr( (coor[i[1]][0] - coor[i[0]][0])*(coor[i[2]][1] - coor[i[0]][1]) - (coor[i[1]][1] - coor[i[0]][1])*(coor[i[2]][0] - coor[i[0]][0]) >= ((-1)*maxi)*(1-z[i[0]]))

A = gu.quicksum(a[i,j] for i in M for j in M if i < len(M)-1)

for i in M:
    for j in M:    
        if i < len(M)-1:
            m.addConstr(a[i,j] >= (coor[i+1][0] - coor[i][0])*(coor[j][1] - coor[i][1]) - (coor[i+1][1] - coor[i][1])*(coor[j][0] - coor[i][0]))
        

m.setObjective(A , sense = gu.GRB.MAXIMIZE)
m.update()
m.setParam('OutputFlag',1)
m.optimize()

print(m.ObjVal)
'''