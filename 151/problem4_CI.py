import scipy.stats as stats

def find_probability(n, threshold):
    # 设定概率 p 的初始范围
    p_min, p_max = 0.1, 1
    precision = 1e-6  # 解的精度
    m = int(n * 0.05)  # 次品数量

    while p_max - p_min > precision:
        p_mid = (p_min + p_max) / 2
        cumulative_prob = sum(stats.binom.pmf(i, n, p_mid) for i in range(m))  # i 从 0 到 m
        if cumulative_prob >= threshold:
            p_min = p_mid
        else:
            p_max = p_mid

    right_bound = p_mid
    p_min, p_max = 0, 0.1
    while p_max - p_min > precision:
        p_mid = (p_min + p_max) / 2
        cumulative_prob = sum(stats.binom.pmf(i, n, p_mid) for i in range(m, n+1))  # i 从 m 到 n
        if cumulative_prob >= threshold:
            p_max = p_mid
        else:
            p_min = p_mid
    
    left_bound = p_mid

    return left_bound, right_bound

n = 100  # 抽样数
threshold = 0.05  # 概率阈值

left, right = find_probability(n, threshold)

print(f"95% CI for n={n} is [{left}, {right}]")
