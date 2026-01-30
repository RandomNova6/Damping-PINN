import numpy as np

def damping_by_log_decrement(amplitudes, periods=None):
    """
    逐差法计算阻尼比
    
    参数:
        amplitudes: 峰值振幅列表 [A0, A1, A2, ...]
        periods: 对应周期列表，可选
    
    返回:
        zeta: 阻尼比
        delta: 对数衰减率
        beta: 衰减系数（如有周期）
    """
    amps = np.array(amplitudes)
    n = len(amps)
    
    if n < 3:
        raise ValueError("至少需要3个峰值")
    
    log_ratios = [np.log(amps[i]/amps[i+1]) for i in range(n-1)]
    delta = np.mean(log_ratios)
    
    zeta = delta / np.sqrt(4*np.pi**2 + delta**2)
    
    beta = None
    if periods is not None:
        periods = np.array(periods)
        avg_period = np.mean(periods[:len(periods)-1])
        beta = delta / avg_period
    
    return zeta, delta, beta

if __name__ == "__main__":
    theta=[130.5,116.0,103.0,92.0,81.5,72.5,64.5,57.0,51.0,45.0]
    omega=2*np.pi/np.array([1.7308,1.7310,1.7312,1.7312,1.7314,1.7317,1.7318,1.7323,1.7331,1.7337])
    zeta, delta, beta = damping_by_log_decrement(theta,omega)
    print(f"阻尼比 zeta: {zeta:.6f}, 对数衰减率 delta: {delta:.6f}, 衰减系数 beta: {beta:.6f}")