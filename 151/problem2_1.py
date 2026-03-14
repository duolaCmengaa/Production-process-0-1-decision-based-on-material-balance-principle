# 问题2非线性0-1整数规划的遗传算法求解

import numpy as np
from scipy.optimize import minimize
from deap import base, creator, tools, algorithms
import pandas as pd
# import matplotlib.pyplot as plt
import openpyxl

# 定义问题中的常数
# 代码重构时考虑了六种情况，将常数保存至params[i]中，写入for循环
'''e1, e2, ep = 0.1, 0.2, 0.1 # 零件与成品的次品率
p1, p2 = 4, 18 # 零件各自的采购价
tc1, tc2, tcp = 8, 1, 2 # 零件与成品的检测成本
ass, dass = 6, 40 # 组装成本与拆解成本
re = 6 # 调换成本'''

# 创建遗传算法的适应度函数，目标是最小化问题
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# 初始化种群的个体，整数变量(0或1)
def init_individual():
    # 整数变量部分：t1, t2, tp, dp, tn1, tn2 (0或1)
    int_vars = np.random.randint(0, 2, 4).tolist()
    return creator.Individual(int_vars)

def objective(solution, t1, t2, t1_, t2_, tp, dp, params): # 目标函数，即成本
    n1, n2, l1, l2, np, e1h, e2h, e1_, e2_, eph = solution
    p1, p2 = params[0:2]
    tc1, tc2, tcp = params[2:5]
    ass, dass = params[5:7]
    re = params[7]
    obj = n1 * p1 + n2 * p2 + \
          n1 * t1 * tc1 + n2 * t2 * tc2 + np * tp * tcp + \
          np * dp * eph * (t1_ * tc1 + t2_ * tc2) + \
          np * ass + \
          np * dp * eph * dass + \
          np * (1 - tp) * eph * re
    return obj

def mse(vars, t1, t2, t1_, t2_, tp, dp, params): # 计算约束方程的均方误差
    n1, n2, l1, l2, np, e1h, e2h, e1_, e2_, eph = vars # 采购流量，拆解流量，总流量，以及次品率
    e1, e2, ep = params
    # 流量约束
    eq1 = n1 * (1 - e1 * t1) + l1 - np
    eq2 = n2 * (1 - e2 * t2) + l2 - np
    eq3 = np * dp * eph * (1 - e1_ * t1_) - l1
    eq4 = np * dp * eph * (1 - e2_ * t2_) - l2
    eq5 = np * (1 - eph) - 1

    # 次品率约束
    eq6 = np * e1h - n1 * e1 * (1 - t1) - l1 * e1_ * (1 - t1_)
    eq7 = np * e2h - n2 * e2 * (1 - t2) - l2 * e2_ * (1 - t2_)
    eq8 = e1h * dp - eph * e1_ * dp
    eq9 = e2h * dp - eph * e2_ * dp
    eq10 = eph - ep * (1 - e1h) * (1 - e2h) - (e1h + e2h - e1h * e2h)

    return sum(x**2 for x in [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10])/10

# 评估目标函数的值
def evaluate(individual, params):
    t1, t2, tp, dp = individual
    t1_, t2_ = 1 - t1, 1 - t2
    # 最小化均方误差mse，以此解得各个非决策变量的数值解
    initial_guess = [0.5 for _ in range(10)]
    solution = minimize(lambda vars: mse(vars, t1, t2, t1_, t2_, tp, dp, params[0:3]), 
                        initial_guess, 
                        options={'maxiter': 3000},
                        tol=1e-6,
                        bounds=[(0, None) for _ in range(5)] +\
                        [(0, 1) for _ in range(5)]) #流量约束为>0, 次品率约束为0-1之间的连续值

    if solution.success:
        vars_solution = solution.x  
        print(f't1: {t1}, t2: {t2}, t1_: {t1_}, t2_: {t2_}, tp: {tp}, dp: {dp}')
        print(f'n1: {vars_solution[0]}, n2: {vars_solution[1]}, l1: {vars_solution[2]}, l2: {vars_solution[3]}, np: {vars_solution[4]}, e1h: {vars_solution[5]}, e2h: {vars_solution[6]}, e1_: {vars_solution[7]}, e2_: {vars_solution[8]}, eph: {vars_solution[9]}')
        print(f'MSE: {mse(vars_solution, t1, t2, t1_, t2_, tp, dp, params[0:3])}')

        # 计算目标函数的值
        objective_value = objective(vars_solution, t1, t2, t1_, t2_, tp, dp, params[3:11])

        print(f'Objective value: {objective_value}')
        return objective_value,

    else:
        return float('inf'),  # 如果优化失败，返回无穷大，快速淘汰此个体

# 注册遗传算法的主要操作
toolbox = base.Toolbox()
toolbox.register("individual", init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxUniform, indpb=0.5)  # 使用均匀交叉

# 为整数变量定义变异操作，确保变异后值仍为0或1
def mutate_integer(individual, indpb):
    for i in range(4):
        if np.random.random() < indpb:
            individual[i] = 1 - individual[i]  # 0变1，1变0
    return individual,

# 注册整数的变异
toolbox.register("mutate", mutate_integer, indpb=0.2)

# 注册选择操作（锦标赛选择）
toolbox.register("select", tools.selTournament, tournsize=3)

# 定义遗传算法的执行步骤
def run_ga(params):

    toolbox.register("evaluate", evaluate, params=params)

    # 设置种群数为100
    population = toolbox.population(n=100)
    # 初始化一个列表来存储每一代的最低值
    min_values_per_generation = []

    # 进行5代进化
    for gen in range(5):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.2)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))
        now_best = tools.selBest(population, k=1)[0]

        # 记录这一代的最低值
        min_value = toolbox.evaluate(now_best)
        min_values_per_generation.append(min_value)
        print(f"Generation {gen}: Min value = {min_value}")

    return population, min_values_per_generation

params = []
params.append([0.1, 0.1, 0.1, 4, 18, 2, 3, 3, 6, 5, 6])
params.append([0.2, 0.2, 0.2, 4, 18, 2, 3, 3, 6, 5, 6])
params.append([0.1, 0.1, 0.1, 4, 18, 2, 3, 3, 6, 5, 30])
params.append([0.2, 0.2, 0.2, 4, 18, 1, 1, 2, 6, 5, 30])
params.append([0.1, 0.2, 0.1, 4, 18, 8, 1, 2, 6, 5, 10])
params.append([0.05, 0.05, 0.05, 4, 18, 2, 3, 3, 6, 40, 10])

# 保存位置
file_path = './result/problem2.xlsx'

book = openpyxl.Workbook()
book.save(file_path)

for i in range(6):
    # 执行遗传算法
    final_population, min_values = run_ga(params[i])

    # 输出最优个体和其目标值
    best_individual = tools.selBest(final_population, k=1)[0]
    print("Best individual:", best_individual)
    print("Objective value:", toolbox.evaluate(best_individual))

    # 将结果保存至excel中
    t = best_individual[:2]
    tp = best_individual[2]
    dp = best_individual[3]
    t_ = [1-t[0], 1-t[1]]

    result_1 = pd.DataFrame({'t': t, 't\'': t_}, index=range(1, 3))
    result_2 = pd.DataFrame({'tp': tp, 'dp': dp}, index=[0])
    result = pd.concat([result_1, result_2])

    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        result.to_excel(writer, sheet_name=f'情况{i+1}')
    

    # 情况较少，易于优化，故不打印优化曲线
    '''plt.plot(min_values)
    plt.show()'''

book = openpyxl.load_workbook(file_path)
del book['Sheet']
book.save(file_path)