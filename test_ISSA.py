# -*- coding: utf-8 -*-
"""
Created on Sat May  7 15:57:14 2022

@author: asus
"""

import numpy as np
import pandas as pd
import function_ISSA as fun
import matplotlib.pyplot as plt

def main(Function_name): # 传入选择的基准测试函数
    SearchAgents_no=100 # 种群数量
    Max_iteration=50 # 最大迭代次数
    [lb,ub,dim]=fun.Parameters(Function_name) # 获取搜索区域范围lb~ub，搜索维度
    [fMin,bestX,SSA_curve]=fun.ISSA(SearchAgents_no,Max_iteration,lb,ub,dim,Function_name)
    
    # print(['最优值为：',fMin])
    # print(['最优变量为：',bestX])
    # thr1=np.arange(len(SSA_curve[0,:]))
    
    # plt.plot(thr1, SSA_curve[0,:])
    
    # plt.xlabel('num')
    # plt.ylabel('object value')
    # plt.title('line')
    # plt.show()
    return fMin
    
if __name__=='__main__':
    # main('F8')
    result=[]
    # 独立运行30次并保存结果至csv
    functions=['F1','F2','F3','F4','F5','F6','F7','F8','F9']
    for fun_name in functions:
        temp=[]
        for _ in range(30):
            temp.append(main(fun_name))
        rmin=np.min(temp)
        rmean=np.mean(temp)
        rstd=np.std(temp)
        result.append([rmin,rmean,rstd])

    result1=pd.DataFrame(data=result,columns=['Min','Mean','Std'],index=functions)
    result1.to_csv(r'C:\Users\asus\Desktop\研一下课程\算法导论\第14-15周-综合实验报告\result_ISSA.csv', encoding='utf-8')
    print('Done!')
