# -*- coding: utf-8 -*-
"""
Created on Thu May  5 22:53:12 2022

@author: asus
"""

import numpy as np

# class function:
#     def __init__(self):
#         print("function init")

# 获取基准函数的范围与维度
def Parameters(F):
    # 单峰函数
    if F=='F1':
        ParaValue = [-100,100,30]
    elif F=='F2':
        ParaValue = [-10, 10, 30]
    elif F=='F3':
        ParaValue = [-100, 100, 30]
    elif F=='F4':
        ParaValue = [-100, 100, 30]
    elif F=='F5':
        ParaValue = [-100,100,30]
    # 多峰函数
    elif F=='F6':
        ParaValue = [-5.12,5.12,30]
    elif F=='F7':
        ParaValue = [-32,32,30]
    elif F=='F8':
        ParaValue = [-50,50,30]
    elif F=='F9':
        ParaValue = [-50,50,30]
    return ParaValue

# 基准测试函数，用于计算个体适应度
def fun(F,X):
    # 单峰函数
    if F=='F1':
        O=np.sum(X*X)
    elif F=='F2':
        O=np.sum(np.abs(X))+np.prod(np.abs(X))
    elif F=='F3':
        O=0
        for i in range(len(X)):
            O=O+np.square(np.sum(X[0:i+1]))
    elif F=='F4':
        O=np.max(np.abs(X))
    elif F=='F5':
        O=np.sum(np.square(np.abs(X+0.5)))
    # 多峰函数
    elif F=='F6':
        O=np.sum(X*X-10*np.cos(2*np.pi*X))+10*len(X)
    elif F=='F7':
        O=-20*np.exp(-0.2*np.power(1/len(X)*np.sum(X*X),0.5))-np.exp(1/len(X)*np.sum(np.cos(2*np.pi*X)))+20+np.e
    elif F=='F8':
        O=0.1*((np.sin(3*np.pi*X[0]))**2+np.sum((X-1)**2*(1+(np.sin(3*np.pi*X[0]+1))**2))+(X[-1]-1)**2*(1+np.sin(2*np.pi*X[-1])**2))+u(X,5,100,4)
    elif F=='F9':
        Y=1+(X+1)/4
        O=np.pi/len(X)*(10*np.sin(np.pi*Y[0])+np.sum((Y[0:len(X)-1])**2*(1+10*(np.sin(np.pi*Y[1:]))**2))+(Y[-1]-1)**2)+u(X,10,100,4)
    return O

# 多峰基准测试函数中的u函数
def u(X,a,k,m):
    u=0
    for x in X:
        if x>=a:
            u+=k*(x-a)**m
        elif -a<x<a:
            continue
        else:
            u+=k*(-x-a)**m
    return u

# 去除超过边界的变量
def Bounds(s,Lb,Ub):
    temp=s
    for i in range(len(s)):
        if temp[i]<Lb[0,i]:
            temp[i]=Lb[0,i]
        elif temp[i]>Ub[0,i]:
            temp[i]=Ub[0,i]
    return temp

# 传入参数：种群数量，最大迭代次数，最小搜索区域范围，最大搜索区域范围，维度，基准测试函数
def ISSA(pop,M,c,d,dim,f):
    # 初始定义
    P_percent=0.2 # 探索者占总种群的比例
    pNum=round(pop*P_percent) # 探索者个数
    sNum=int(np.ceil(pop*0.2)) # 预警者个数
    wmax=3 # 正弦搜索策略的最大范围
    wmin=1 # 正弦搜索策略的最小范围
    
    lb=c*np.ones((1,dim))
    ub=d*np.ones((1,dim))
    X=np.zeros((pop,dim)) 
    fit=np.zeros((pop,1))
    Convergence_curve=np.zeros((1,M)) # 初始化收敛曲线
    
    # 随机初始化种群位置和适应度
    for i in range(pop):
        X[i,:]=lb+(ub-lb)*np.random.rand(1,dim) # rand：服从均匀分布，创建[0,1)之间随机浮点数
        fit[i,0]=fun(f,X[i,:])
    
    pFit=fit # 初始化最佳适应度矩阵
    pX=X # 初始化最佳种群位置
    fMin=np.min(fit[:,0]) # 初始化当前全局最优适应度值，生产者能量储备水平取决于对个人适应度值的评估
    bestI=np.argmin(fit[:,0])
    bestX=X[bestI,:] # 初始化当前全局最优位置
    
    # 迭代
    for t in range(M):
        sortIndex=np.argsort(pFit.T) # 转置矩阵后获取排序后各适应度对应的索引
        fmax=np.max(pFit[:,0]) # 当前全局最差适应度
        B=np.argmax(pFit[:,0]) 
        worse=X[B,:] # 当前全局最差位置
        
        # 探索者位置更新算法
        r2=np.random.rand(1) # 发现捕食者的个体所发出的示警信号值
        if r2 < 0.8: # 预警值较小，说明没有捕食者出现
            for i in range(pNum):
                r1=np.random.rand(1)
                X[sortIndex[0,i],:]=pX[sortIndex[0,i],:]*np.exp(-(i)/(r1*M))
                X[sortIndex[0,i],:]=Bounds(X[sortIndex[0,i],:],lb,ub)
                fit[sortIndex[0,i],0]=fun(f,X[sortIndex[0,i],:])
        elif r2 >= 0.8: # 预警值较大，说明有捕食者出现威胁到了种群的安全，需要去其它地方觅食
            for i in range(pNum):
                w=wmin+(wmax-wmin)*(np.sin(((pFit[i,0]-fMin)/(fmax-fMin)+1)*np.pi/2+np.pi)+1) # 引入正弦搜索策略，下同
                Q = np.random.randn(1)
                X[sortIndex[0,i],:]=pX[sortIndex[0,i],:]+w*Q*np.ones((1,dim)) 
                X[sortIndex[0,i],:]=Bounds(X[sortIndex[0,i],:],lb,ub)
                fit[sortIndex[0,i],0]=fun(f,X[sortIndex[0,i],:])
        bestII=np.argmin(fit[:,0])
        bestXX=X[bestII,:] # 更新探索者当前全局最优位置
        
        # 追随者位置更新算法
        for ii in range(pop-pNum):
            i=ii+pNum
            if i > pop/2: # 这部分追随者处于十分饥饿的状态（因为它们的能量很低，也就是适应度值很差），需要到其它地方觅食
                Q = np.random.randn(1)
                X[sortIndex[0,i],:]=Q*np.exp(worse-pX[sortIndex[0,i],:]/np.square(i))
            else:  # 这部分追随者是围绕能量最高的发现者周围进行觅食，其间也有可能发生食物的争夺，使其自己变成探索者
                w=wmin+(wmax-wmin)*(np.sin(((pFit[i,0]-fMin)/(fmax-fMin)+1)*np.pi/2+np.pi)+1)
                A=np.floor(np.random.rand(1,dim)*2)*2-1 # 为A中元素随机赋值1或-1。floor: 只取整数部分
                X[sortIndex[0,i],:]=bestXX+w*np.dot(np.abs(pX[sortIndex[0,i],:]-bestXX),1/(A.T*np.dot(A,A.T)))*np.ones((1,dim))
            X[sortIndex[0,i],:]=Bounds(X[sortIndex[0,i],:],lb,ub)
            fit[sortIndex[0,i],0]=fun(f,X[sortIndex[0,i],:])
        
        # 预警者位置更新算法（注意：这里只是意识到了危险，不代表出现了真正的捕食者）
        arrc = np.arange(len(sortIndex[0,:]))
        c=np.random.permutation(arrc) # 在种群中随机产生位置
        b=sortIndex[0,c[0:sNum]] # 取前20%个麻雀
        for j in range(len(b)):
            w=wmin+(wmax-wmin)*(np.sin(((pFit[i,0]-fMin)/(fmax-fMin)+1)*np.pi/2+np.pi)+1)
            if pFit[sortIndex[0,b[j]],0]>fMin: # 麻雀处于种群的边缘，极易受到捕食者的攻击，会向安全区域靠拢
                X[sortIndex[0,b[j]],:]=bestX+w*np.random.randn(1,dim)*np.abs(pX[sortIndex[0,b[j]],:]-bestX)
            else: # 麻雀处于种群中心，会随机行走以靠近别的麻雀
                X[sortIndex[0,b[j]],:]=pX[sortIndex[0,b[j]],:]+w*(2*np.random.rand(1)-1)*np.abs(pX[sortIndex[0,b[j]],:]-worse)/(pFit[sortIndex[0,b[j]]]-fmax+10**(-50))
            X[sortIndex[0,b[j]],:]=Bounds(X[sortIndex[0,b[j]],:],lb,ub)
            fit[sortIndex[0,b[j]],0]=fun(f,X[sortIndex[0,b[j]]])
            
        # 多样性变异处理，仅在迭代初期使用
        mean=np.mean(pFit.T)
        var=np.cov(pFit.T)
        A=(var-mean)/mean**2 # 种群聚集度指标
        if t<=M/2 and A>0.01: # A>1表示种群为聚集状态，否则为随机状态
            n=np.tan((np.random.rand(1)-0.5)*np.pi)
            bestXnew=bestX+bestX*0.5/np.pi/(n**2+0.5**2)
            bestX=bestXnew
        
        # 更新当前全局最优适应值、全局最优位置（即比较变异后个体是否优于原个体），并生成收敛曲线值
        for i in range(pop):
            if fit[i,0]<pFit[i,0]:
                pFit[i,0]=fit[i,0]
                pX[i,:]=X[i,:]
            if pFit[i,0]<fMin:
                fMin=pFit[i,0]
                bestX=pX[i,:]
        Convergence_curve[0,t]=fMin
        
    # 返回参数：最优值，最优变量，收敛曲线值
    return fMin,bestX,Convergence_curve






