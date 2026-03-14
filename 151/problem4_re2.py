# 问题4的敏感性分析 重做问题2
# 6个决策变量分别置为0或1，总共是2^6种情况，因此可以采用枚举进行求解

import itertools
from scipy.optimize import minimize

# 定义问题中的常数
# 更改零件1的次品率为置信区间上界
e1, e2, ep = 0.266, 0.2, 0.2 # 零件与成品的次品率
p1, p2 = 4, 18 # 零件各自的采购价
tc1, tc2, tcp = 2, 3, 3 # 零件与成品的检测成本
ass, dass = 6, 5 # 组装成本与拆解成本
re = 6 # 调换成本

# 定义目标函数，即成本
def objective(solution, t1, t2, t1_, t2_, tp, dp):
    n1, n2, l1, l2, np, e1h, e2h, e1_, e2_, eph = solution
    obj = n1 * p1 + n2 * p2 + \
          n1 * t1 * tc1 + n2 * t2 * tc2 + np * tp * tcp + \
          np * dp * eph * (t1_ * tc1 + t2_ * tc2) + \
          np * ass + \
          np * dp * eph * dass + \
          np * (1 - tp) * eph * re
    return obj

# 定义 MSE 方程，用于求解非决策变量
def mse(vars, t1, t2, t1_, t2_, tp, dp):
    n1, n2, l1, l2, np, e1h, e2h, e1_, e2_, eph = vars

    # 流量约束方程
    eq1 = n1 * (1 - e1 * t1) + l1 - np
    eq2 = n2 * (1 - e2 * t2) + l2 - np
    eq3 = np * dp * eph * (1 - e1_ * t1_) - l1
    eq4 = np * dp * eph * (1 - e2_ * t2_) - l2
    eq5 = np * (1 - eph) - 1

    # 次品率约束方程
    eq6 = np * e1h - n1 * e1 * (1 - t1) - l1 * e1_ * (1 - t1_)
    eq7 = np * e2h - n2 * e2 * (1 - t2) - l2 * e2_ * (1 - t2_)
    eq8 = e1h * dp - eph * e1_ * dp
    eq9 = e2h * dp - eph * e2_ * dp
    eq10 = eph - ep * (1 - e1h) * (1 - e2h) - (e1h + e2h - e1h * e2h)

    return sum(x**2 for x in [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10])/10

# 开始枚举
combinations = list(itertools.product([0, 1], repeat=6))
initial_guess = [0.5 for _ in range(10)]

min_obj = 0
i = 0
for choice in combinations:
    t1, t2, t1_, t2_, tp, dp = choice
    # 跳过不满足决策变量约束的解，只枚举满足约束的解
    if (t1 + t1_ - 1) != 0 or (t2 + t2_ - 1) != 0:
        continue 
    # 调用 minimize，并传入约束条件列表
    solution = minimize(lambda vars: mse(vars, t1, t2, t1_, t2_, tp, dp), 
                        initial_guess, 
                        options={'maxiter': 5000},
                        tol=1e-6,
                        bounds=[(0, None) for _ in range(5)] + [(0, 1) for _ in range(5)])

    # 获取结果并检查
    print(i)
    print(solution.message)
    if solution.success:
        vars_solution = solution.x  # 从优化结果中提取解
        print(f't1: {t1}, t2: {t2}, t1_: {t1_}, t2_: {t2_}, tp: {tp}, dp: {dp}')
        print(f'n1: {vars_solution[0]}, n2: {vars_solution[1]}, l1: {vars_solution[2]}, l2: {vars_solution[3]}, np: {vars_solution[4]}, e1h: {vars_solution[5]}, e2h: {vars_solution[6]}, e1_: {vars_solution[7]}, e2_: {vars_solution[8]}, eph: {vars_solution[9]}')
        print(f'MSE: {mse(vars_solution, t1, t2, t1_, t2_, tp, dp)}')

        obj = objective(vars_solution, t1, t2, t1_, t2_, tp, dp) # 计算目标函数
        print(f'Objective: {obj}')

        # 更新最优解
        if i == 0 or obj < min_obj:
            min_obj = obj
            best_choice = choice
            best_solution = vars_solution
    i += 1

# 输出最优结果
print('\n最优结果：')
print(f't1: {best_choice[0]}, t2: {best_choice[1]}, t1_: {best_choice[2]}, t2_: {best_choice[3]}, tp: {best_choice[4]}, dp: {best_choice[5]}')
print(f'min_obj: {min_obj}')
print(f'n1: {best_solution[0]}, n2: {best_solution[1]}, l1: {best_solution[2]}, l2: {best_solution[3]}, np: {best_solution[4]}, e1h: {best_solution[5]}, e2h: {best_solution[6]}, e1_: {best_solution[7]}, e2_: {best_solution[8]}, eph: {best_solution[9]}')

# 可以自行改变情况常数进行求解复现