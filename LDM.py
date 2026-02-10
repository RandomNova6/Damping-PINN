import numpy as np

def damping_by_successive_diff(amplitudes, periods=None):
    """
    使用【真正的逐差法】计算阻尼参数
    
    参数:
        amplitudes: 峰值振幅列表 [A0, A1, A2, ...]
        periods: 对应周期列表 [T1, T2, ...]，单位为秒(s)。注意：不是角频率！
    
    返回:
        zeta: 阻尼比 (Damping Ratio)
        delta: 对数衰减率 (Logarithmic Decrement)
        beta: 衰减系数 (Damping Coefficient, 1/s)
    """
    amps = np.array(amplitudes)
    n = len(amps)
    
    if n < 4:
        raise ValueError("逐差法建议至少需要4个峰值数据")
    
    # --- 1. 真正的逐差法计算 delta ---
    # 将数据分为前后两半
    ln_amps = np.log(amps)
    k = n // 2  # 间隔跨度 (半数)
    
    # 逐差法公式: (前一半的对数和 - 后一半的对数和)
    # 相当于 sum(ln(A_i) - ln(A_{i+k}))
    # 理论上 ln(A_i) - ln(A_{i+k}) = k * delta
    sum_diff = 0
    for i in range(k):
        # 对应项相减：ln(A_i) - ln(A_{i+k})
        diff = ln_amps[i] - ln_amps[i+k]
        sum_diff += diff
        
    # 计算 delta
    # 公式推导: mean_diff = k * delta => delta = sum_diff / (k * k)
    # 如果数据是偶数个，正好分成两组 k个差值。
    # 如果是奇数个，中间项通常不用或归入某一组，这里采用简单的 k = n//2 舍弃末尾或中间方式
    delta = sum_diff / (k * k)
    
    # --- 2. 计算阻尼比 zeta ---
    # 公式: zeta = delta / sqrt(4*pi^2 + delta^2)
    zeta = delta / np.sqrt(4*np.pi**2 + delta**2)
    
    # --- 3. 计算衰减系数 beta ---
    beta = None
    if periods is not None:
        periods = np.array(periods)
        # 这里的输入必须是周期T(s)，不是角频率
        avg_period = np.mean(periods)
        # beta = delta / T
        beta = delta / avg_period
    
    return zeta, delta, beta

if __name__ == "__main__":
    # 振幅数据
    theta = [130.5, 116.0, 103.0, 92.0, 81.5, 72.5, 64.5, 57.0, 51.0, 45.0]
    
    # 原始数据看起来是周期 T (秒)，而不是频率
    # 假设题目给的 [1.7308...] 是周期 T
    raw_periods = np.array([1.7308, 1.7310, 1.7312, 1.7312, 1.7314, 1.7317, 1.7318, 1.7323, 1.7331, 1.7337])
    
    # 注意：直接传入周期 T，不要转换成 omega
    zeta, delta, beta = damping_by_successive_diff(theta, raw_periods)
    
    print("-" * 30)
    print("修正后的逐差法计算结果：")
    print(f"对数衰减率 delta: {delta:.6f}")
    print(f"阻尼比     zeta : {zeta:.6f}")
    print(f"衰减系数   beta : {beta:.6f} (1/s)")
    print("-" * 30)
    
    # 验证计算
    # 理论上 beta 也约等于 zeta * (2*pi / T_avg)
    omega_mean = 2 * np.pi / np.mean(raw_periods)
    print(f"验算: zeta * omega_n ≈ {zeta * omega_mean:.6f}")