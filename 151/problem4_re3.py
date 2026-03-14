# 问题4 敏感度分析 重做问题3
# 单独改变某一零件、半成品、成品的次品率（置为置信区间的临界值），观察是否对决策产生影响
# 在定义问题中的常数处修改以复现我们的结果

import numpy as npy
from scipy.optimize import minimize
from deap import base, creator, tools, algorithms
import pandas as pd
import matplotlib.pyplot as plt
import time 

time1 = time.time()
# 定义问题中的常数
e = [0.1 for _ in range(8)]      # 8个零件的次品率
p = [2, 8, 12, 2, 8, 12, 8, 12]  # 8个零件的采购价
tc = [1, 1, 2, 1, 1, 2, 1, 2]    # 8个零件的检测价
eps = [0.1 for _ in range(3)]    # 半成品装配的次品率
ep = 0.055                       # 成品装配的次品率
assps = [8, 8, 8]                # 半成品的加工费
assp = 8                         # 成品的加工费
tcps = [4, 4, 4]                 # 半成品的检测价
tcp = 6                          # 成品的检测价
dassps = [6, 6, 6]               # 半成品的拆解费
dassp = 10                       # 成品的拆解费
re = 40                          # 成品调换损失

# 创建遗传算法的适应度函数，目标是最小化问题
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# 初始化种群的个体，整数变量(0或1)
def init_individual():
    # 整数变量部分：19个决策变量
    int_vars = npy.random.randint(0, 2, 19).tolist()
    return creator.Individual(int_vars)

# 定义目标函数(成本)的计算
def objective(solution, t1, t2, t3, t4, t5, t6, t7, t8, 
               t1_, t2_, t3_, t4_, t5_, t6_, t7_, t8_, 
               t1__, t2__, t3__, t4__, t5__, t6__, t7__, t8__,
               tp1, tp2, tp3, tp, 
               tp1_, tp2_, tp3_, 
               dp1, dp2, dp3, dp,
               dp1_, dp2_, dp3_
               ):
    t = [t1, t2, t3, t4, t5, t6, t7, t8]
    t_ = [t1_, t2_, t3_, t4_, t5_, t6_, t7_, t8_]
    t__ = [t1__, t2__, t3__, t4__, t5__, t6__, t7__, t8__]
    tps = [tp1, tp2, tp3]
    tps_ = [tp1_, tp2_, tp3_]
    dps = [dp1, dp2, dp3]
    dps_ = [dp1_, dp2_, dp3_]

    n1, n2, n3, n4, n5, n6, n7, n8, \
    l1, l2, l3, l4, l5, l6, l7, l8, \
    l1_, l2_, l3_, l4_, l5_, l6_, l7_, l8_, \
    np1, np2, np3, np, \
    lp1, lp2, lp3, \
    e1h, e2h, e3h, e4h, e5h, e6h, e7h, e8h, \
    e1hh, e2hh, e3hh, e4hh, e5hh, e6hh, e7hh, e8hh, \
    e1_, e2_, e3_, e4_, e5_, e6_, e7_, e8_, \
    e1__, e2__, e3__, e4__, e5__, e6__, e7__, e8__, \
    ep1h, ep2h, ep3h, eph, \
    ep1hh, ep2hh, ep3hh, \
    ep1_, ep2_, ep3_ = solution

    n = [n1, n2, n3, n4, n5, n6, n7, n8]
    l = [l1, l2, l3, l4, l5, l6, l7, l8]
    nps = [np1, np2, np3]
    lps = [lp1, lp2, lp3]
    eh = [e1h, e2h, e3h, e4h, e5h, e6h, e7h, e8h]
    ehh = [e1hh, e2hh, e3hh, e4hh, e5hh, e6hh, e7hh, e8hh]
    e_ = [e1_, e2_, e3_, e4_, e5_, e6_, e7_, e8_]
    ephs = [ep1h, ep2h, ep3h]
    eps_ = [ep1_, ep2_, ep3_]

    obj = 0

    for i in range(8):
        obj += n[i] * p[i]                 # 采购费用
        obj += n[i] * tc[i] * t[i]         # 采购的零件是否检测产生的费用
    for i in range(3):
        obj += nps[i] * tps[i] * tcps[i]   # 半成品是否检测产生的费用
        obj += nps[i] * assps[i]           # 半成品的装配费用

        # 半成品是否拆解产生的费用
        obj += (nps[i] * tps[i] * dps[i] * ephs[i] + np * eph * dp * eps_[i] * tps_[i] * dps_[i]) * dassps[i]

    obj += np * tp * tcp   # 成品是否检测产生的费用                 

    # 拆解后回流的零件是否检测产生的费用
    obj += np1 * tp1 * dp1 * ep1h * (t1_*tc[0] + t2_*tc[1] + t3_*tc[2]) 
    obj += np2 * tp2 * dp2 * ep2h * (t4_*tc[3] + t5_*tc[4] + t6_*tc[5])
    obj += np3 * tp3 * dp3 * ep3h * (t7_*tc[6] + t8_*tc[7])

    obj += np * eph * dp * ep1_ * tp1_ * dp1_ * (t1__ * tc[0] + t2__ * tc[1] + t3__ * tc[2])
    obj += np * eph * dp * ep2_ * tp2_ * dp2_ * (t4__ * tc[3] + t5__ * tc[4] + t6__ * tc[5])
    obj += np * eph * dp * ep3_ * tp3_ * dp3_ * (t7__ * tc[6] + t8__ * tc[7])

    # 拆解后回流的半成品是否检测产生的费用
    obj += np * dp * eph * (tp1_*tcps[0] + tp2_*tcps[1] + tp3_*tcps[2])

    # 成品的装配费用与是否拆解成品产生的费用
    obj += np * assp
    obj += np * dp * eph * dassp

    # 是否产生调换损失
    obj += np * (1-tp) * eph * re

    return obj

# 定义方程
def mse(vars, t1, t2, t3, t4, t5, t6, t7, t8, 
               t1_, t2_, t3_, t4_, t5_, t6_, t7_, t8_, 
               t1__, t2__, t3__, t4__, t5__, t6__, t7__, t8__,
               tp1, tp2, tp3, tp, 
               tp1_, tp2_, tp3_, 
               dp1, dp2, dp3, dp,
               dp1_, dp2_, dp3_
               ):
    
    # 决策变量
    t = [t1, t2, t3, t4, t5, t6, t7, t8]
    t_ = [t1_, t2_, t3_, t4_, t5_, t6_, t7_, t8_]
    t__ = [t1__, t2__, t3__, t4__, t5__, t6__, t7__, t8__]
    tps = [tp1, tp2, tp3]
    tps_ = [tp1_, tp2_, tp3_]
    dps = [dp1, dp2, dp3]
    dps_ = [dp1_, dp2_, dp3_]

    # 非决策变量
    n1, n2, n3, n4, n5, n6, n7, n8, \
    l1, l2, l3, l4, l5, l6, l7, l8, \
    l1_, l2_, l3_, l4_, l5_, l6_, l7_, l8_, \
    np1, np2, np3, np, \
    lp1, lp2, lp3, \
    e1h, e2h, e3h, e4h, e5h, e6h, e7h, e8h, \
    e1hh, e2hh, e3hh, e4hh, e5hh, e6hh, e7hh, e8hh, \
    e1_, e2_, e3_, e4_, e5_, e6_, e7_, e8_, \
    e1__, e2__, e3__, e4__, e5__, e6__, e7__, e8__, \
    ep1h, ep2h, ep3h, eph, \
    ep1hh, ep2hh, ep3hh, \
    ep1_, ep2_, ep3_ = vars

    n = [n1, n2, n3, n4, n5, n6, n7, n8]
    l = [l1, l2, l3, l4, l5, l6, l7, l8]
    l_ = [l1_, l2_, l3_, l4_, l5_, l6_, l7_, l8_]
    nps = [np1, np2, np3]
    lps = [lp1, lp2, lp3]
    eh = [e1h, e2h, e3h, e4h, e5h, e6h, e7h, e8h]
    ehh = [e1hh, e2hh, e3hh, e4hh, e5hh, e6hh, e7hh, e8hh]
    e_ = [e1_, e2_, e3_, e4_, e5_, e6_, e7_, e8_]
    e__ = [e1__, e2__, e3__, e4__, e5__, e6__, e7__, e8__]
    ephs = [ep1h, ep2h, ep3h]
    ephhs = [ep1hh, ep2hh, ep3hh]
    eps_ = [ep1_, ep2_, ep3_]

    equations = []

    # 总流量方程
    for i in range(3):
        for j in range(3):
            k = i * 3 + j
            if not k == 8:
                equations.append(n[k] * (1 - e[k] * t[k]) + l[k] + l_[k] - nps[i])
                equations.append(nps[i] * tps[i] * dps[i] * ephs[i] * (1 - e_[k] * t_[k]) - l[k])
                equations.append(l_[k] - dps_[i] * (np * eph - lps[i]) * (1 - t__[k] * e__[k]))
        equations.append(nps[i] * (1 - ephs[i] * tps[i]) + lps[i] - np)
        equations.append(np * dp * eph * (1 - eps_[i] * tps_[i]) - lps[i])
    equations.append(np * (1 - eph) - 1)

    # 次品率方程
    for i in range(3):
        for j in range(3):
            k = i * 3 + j
            if not k==8:
                equations.append(nps[i] * eh[k] - n[k] * e[k] * (1 - t[k]) - l[k] * e_[k] * (1 - t_[k]) - l_[k] * e__[k] * (1 - t__[k]))
                equations.append(eh[k] * tps[i] * dps[i] - tps[i] * dps[i] * ephs[i] * e_[k])
                equations.append(eh[k] * nps[i] * (1 - tps[i]) + ehh[k] * np * eph * dp * (1 - tps_[i]) - np * ehh[k])
                equations.append(ehh[k] * dp * tps_[i] * dps_[i] - eph * dp * eps_[i] * tps_[i] * dps_[i] * e__[k])
        equations.append(np * ephhs[i] - nps[i] * ephs[i] * (1 - tps[i]) - lps[i] * eps_[i] * (1 - tps_[i]))
        equations.append(ephhs[i] * dp - eph * eps_[i] * dp)
    equations.append(ep1h - eps[0] * (1 - e1h) * (1 - e2h) * (1 - e3h) - (e1h + e2h + e3h - e1h * e2h - e1h * e3h - e2h * e3h + e1h * e2h * e3h))
    equations.append(ep2h - eps[1] * (1 - e4h) * (1 - e5h) * (1 - e6h) - (e4h + e5h + e6h - e4h * e5h - e4h * e6h - e5h * e6h + e4h * e5h * e6h))
    equations.append(ep3h - eps[2] * (1 - e7h) * (1 - e8h) - (e7h + e8h - e7h * e8h))
    equations.append(eph - ep * (1 - ep1hh) * (1 - ep2hh) * (1 - ep3hh) - (ep1hh + ep2hh + ep3hh - ep1hh * ep2hh - ep1hh * ep3hh - ep2hh * ep3hh + ep1hh * ep2hh * ep3hh))

    return sum(x**2 for x in equations)/73

# 定义目标函数的评估
def evaluate(individual):

    # 从遗传算法的输出中解码决策变量
    t = individual[:8]
    ones1 = npy.ones_like(t)
    t_ = ones1 - t      # 考虑决策变量的约束
    t__ = ones1 - t
    tps = individual[8:11]
    ones2 = npy.ones_like(tps)
    tp = individual[11]
    tps_ = ones2 - tps  # 考虑决策变量的约束
    dps = individual[12:15]
    dp = individual[15]
    dps_ = individual[16:19]

    initial_guess = [1 for _ in range(8)] + [0.5 for _ in range(65)]

    # 通过最小化mse解得非决策变量方程的解，即非决策变量的值
    solution = minimize(lambda vars: mse(vars, *t, *t_, *t__, *tps, tp, *tps_, *dps, dp, *dps_), 
                        initial_guess, 
                        options={'maxiter': 10000, 'maxfun': 30000},
                        tol=1e-6,

                        # 方程求解的限制，采购流量为>1的连续值，其余流量>0，而次品率为(0，1)间的连续值
                        bounds=[(1, None)for i in range(8)] + [(0, None) for _ in range(23)] + [(0, 1) for _ in range(42)],
                        )

    if solution.success and mse(solution.x, *t, *t_, *t__, *tps, tp, *tps_, *dps, dp, *dps_) < 1e-5:  # 保证方程求解是准确的
        vars_solution = solution.x  # 从优化结果中提取解
        print(f't: {t}, t_: {t_}, t__: {t__}, tps: {tps}, tp:{tp}, tps_: {tps_}, dps: {dps}, dp:{dp}, dps_:{dps_}')
        print(f'n1: {vars_solution[0]}, n2: {vars_solution[1]}, n3: {vars_solution[2]}, n4: {vars_solution[3]}, n5: {vars_solution[4]}, n6: {vars_solution[5]}, n7: {vars_solution[6]}, n8: {vars_solution[7]}')
        print(f'MSE: {mse(vars_solution, *t, *t_,*t__, *tps, tp, *tps_, *dps, dp, *dps_)}')

        # 计算得到目标函数
        obj = objective(vars_solution, *t, *t_,*t__, *tps, tp, *tps_, *dps, dp, *dps_)
        print('obj:', obj)
        return obj,

    else:
        return float('inf'),  # 如果优化失败，返回无穷大

# 注册遗传算法的主要操作
toolbox = base.Toolbox()
toolbox.register("individual", init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxUniform, indpb=0.5)  # 使用均匀交叉

# 为整数变量定义变异操作，确保变异后值仍为0或1
def mutate_integer(individual, indpb):
    for i in range(19):
        if npy.random.random() < indpb:
            individual[i] = 1 - individual[i]  # 0变1，1变0
    return individual,

# 注册整数的变异
toolbox.register("mutate", mutate_integer, indpb=0.2)

# 注册选择操作（锦标赛选择）
toolbox.register("select", tools.selTournament, tournsize=3)

# 定义遗传算法的执行步骤
def run_ga():
    # 设置种群数为100
    population = toolbox.population(n=100)
    # 初始化一个列表来存储每一代的最低值
    min_values_per_generation = []

    # 进行40代进化
    for gen in range(40):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.2)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))
        now_best = tools.selBest(population, k=1)[0]
        # 记录这一代的最低值
        min_value = evaluate(now_best)
        min_values_per_generation.append(min_value)
        print(f"Generation {gen}: Min value = {min_value}")

    return population, min_values_per_generation

# 执行遗传算法
final_population, min_values = run_ga()
time2 = time.time()

print('running time:', time2 - time1)

# 输出最优个体和其目标值
best_individual = tools.selBest(final_population, k=1)[0]
print("Best individual:", best_individual)
print("Objective value:", evaluate(best_individual))

# 将结果保存至excel中
t = best_individual[:8]
ones1 = npy.ones_like(t)
t_ = ones1 - t      
t__ = ones1 - t
tps = best_individual[8:11]
ones2 = npy.ones_like(tps)
tp = best_individual[11]
tps_ = ones2 - tps  
dps = best_individual[12:15]
dp = best_individual[15]
dps_ = best_individual[16:19]

result_1 = pd.DataFrame({'t': t, 't\'': t_, 't\'\'': t__}, index=range(1, 9))
result_2 = pd.DataFrame({'tps': tps, 'tps\'': tps_, 'dps': dps, 'dps\'': dps_}, index=range(1, 4))
result_3 = pd.DataFrame({'tp': tp, 'dp': dp}, index=[0])
result = pd.concat([result_1, result_2, result_3])
result.to_excel('./result/problem4.xlsx')


# 绘制优化曲线
plt.plot(min_values)
plt.title('Every generation’s optimal individual (decision) objective function (cost).')
plt.xlabel('generational number')
plt.ylabel('cost')
plt.show()