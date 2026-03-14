import pandas as pd
import numpy as np
import os
from math import comb

def get_prob(N, n, p = 0.1):
    """
    获取一批大小为N的产品中有n个次品的概率
    N: 产品数量
    n: 次品数量
    p: 产品为次品的概率，默认为0.1
    """
    prob = comb(N, n) * (p ** n) * ((1 - p) ** (N - n))
    return prob

def generate_matrix(t, sign):
    # 保存Pi(X=j)的概率
    arr = np.zeros((t, t))
    for i in range(1, t):
        for j in range(i + 1):
            arr[i, j] = get_prob(i, j, 0.1)

    cumulative_arr = arr.copy()
    if sign == 1:
        # H0: 零配件次品率低于标称值, H1: 零配件次品率高于标称值
        for i in range(1, t):
            for j in range(i - 1, -1, -1):  # 第一小问，从右往左累加概率得到Pi(X>=j)
                cumulative_arr[i, j] += cumulative_arr[i, j + 1]

    elif sign == 2:
        # H0: 零配件次品率高于标称值, H1: 零配件次品率低于标称值
        for i in range(1, t):
            for j in range(1, i + 1):  # 第二小问，从左往右累加概率得到Pi(X<=j)
                cumulative_arr[i, j] += cumulative_arr[i, j - 1]

    # 转换为dataframe
    df_cumulative = pd.DataFrame(cumulative_arr)
    df_cumulative.columns = [f'{i}' for i in range(t)]
    df_cumulative.index = [f'{i}' for i in range(t)]

    # 保存至excel中
    mkdir = 'result'
    if not os.path.exists(mkdir):
        os.makedirs(mkdir)

    file_name = f"./result/problem1_{sign}.xlsx"
    df_cumulative.to_excel(file_name, index=True)

    print(f"DataFrame has been saved to {file_name}")

    print(df_cumulative)

generate_matrix(10, 1) # 生成情形1结果
generate_matrix(30, 2) # 生成情形2结果