import numpy as np
import pandas as pd

def step_gd(points,learning_rate,m,c):
    N = len(points)
    m_slope=0
    c_slope=0
    for i in range(N):
        x = points[i,0]
        y = points[i,1]
        m_slope += (-2/N)*(y-m*x-c)*x
        c_slope += (-2/N)*(y-m*x-c)
    new_m = m - learning_rate * m_slope
    new_c = c - learning_rate * c_slope
    return new_m,new_c

def cost(points,m,c):
    t_cost=0
    N = len(points)
    for i in range(N):
        x = points[i,0]
        y = points[i,1]
        t_cost = (1/N)*(y-m*x-c)**2
       
    return t_cost

def gd(points,learning_rate,num_iteration):
    m=0
    c=0
    for i in range(num_iteration):
        m,c = step_gd(points,learning_rate,m,c)
        print(i,"cost : ",cost(points,m,c))
    return m,c

def run():
    data = np.loadtxt('/Users/Anurag/Desktop/pythonprojects/gradientdescent/train_boston_gd.csv',delimiter=",")
    learning_rate=0.00001
    num_iteration=100
    m,c = gd(data,learning_rate,num_iteration)
    print(m,c)