import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import os

# ==========================================
# 第一部分：PINN 模型定义 (基于 PyTorch)
# ==========================================

class ForceViberationModel(nn.Module):
    def __init__(self, max_theta_data, omega_peak_data):
        super().__init__()
        
        # --- 神经网络结构 ---
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2)  # 输出: [幅值, 相位]
        )
        
        # --- 物理参数初始化 ---
        self.J_fixed = 0.2
        # 根据数据峰值位置估算 k
        real_k = self.J_fixed * (omega_peak_data ** 2)
        # 初始 c 设为临界阻尼的 5% (保证有明显的尖峰)
        real_c = 0.05 * (2 * np.sqrt(self.J_fixed * real_k))
        # 根据物理公式推算 M0: Theta_max = M0 / (c * omega_peak)
        real_M0 = max_theta_data * real_c * omega_peak_data

        print(f"[PINN Init] 设定 k = {real_k:.5f}, c = {real_c:.5f}, M0 = {real_M0:.5f}")
        
        self.c = nn.Parameter(torch.tensor([real_c], dtype=torch.float32))
        self.k = nn.Parameter(torch.tensor([real_k], dtype=torch.float32))
        self.M0 = nn.Parameter(torch.tensor([real_M0], dtype=torch.float32))

    def physic_model(self, omega):
        # 保证参数非负
        J = self.J_fixed
        c = torch.nn.functional.softplus(self.c)
        k = torch.nn.functional.softplus(self.k)
        M0 = torch.nn.functional.softplus(self.M0)

        omega_val = omega.squeeze()
        omega_sq = omega_val**2

        # --- 振幅公式 ---
        denom_sq = (k - J * omega_sq)**2 + (c * omega_val)**2
        denomin = torch.sqrt(denom_sq + 1e-12)
        theta_phy = M0 / denomin
        
        # --- 相位公式 (标准物理相位 0 -> pi) ---
        phi_phy = torch.atan2(c * omega_val, k - J * omega_sq)
        
        # 计算辅助指标
        omega0 = torch.sqrt(k / J)
        delta = c / (2 * torch.sqrt(J * k)) # 阻尼比 zeta
        
        return {
            'theta': theta_phy.unsqueeze(1),
            'phi': phi_phy.unsqueeze(1),
            'omega0': omega0,
            'delta': delta,
            'params': (J, c, k, M0)
        }

    def forward(self, omega):
        output = self.net(omega)
        theta = torch.nn.functional.softplus(output[:, 0])
        phi = output[:, 1] 
        return {'theta': theta.unsqueeze(1), 'phi': phi.unsqueeze(1)}

class PINNLoss():
    def __init__(self, omega_peak_est):
        self.mse = nn.MSELoss()
        self.omega_target = torch.tensor([omega_peak_est], dtype=torch.float32)

    def __call__(self, model, omega, pred, target, phase="pretrain"):
        # 1. 数据拟合 Loss
        loss_theta_data = self.mse(pred['theta'], target['theta'])
        # 相位改用余弦损失，防止 0/pi 翻转导致的直线拟合
        loss_phi_data = torch.mean(1 - torch.cos(pred['phi'] - target['phi']))
        
        data_total = 30.0 * loss_theta_data + 10.0 * loss_phi_data
        
        if phase == "pretrain":
            return data_total

        # 2. 物理一致性 Loss
        phy_out = model.physic_model(omega)
        loss_theta_phy = self.mse(pred['theta'], phy_out['theta'])
        loss_phi_phy = torch.mean(1 - torch.cos(pred['phi'] - phy_out['phi']))
        
        # 频率对齐
        loss_omega0 = self.mse(phy_out['omega0'], self.omega_target.to(omega.device))
        
        # 3. 阻尼惩罚 (防止阻尼过大导致曲线变平)
        reg_loss = torch.relu(phy_out['delta'] - 0.2) + torch.relu(0.01 - phy_out['delta'])

        return data_total + 350.0 * (loss_theta_phy + loss_phi_phy) + 15.0 * loss_omega0 + 10.0 * reg_loss

class PINNTrainer():
    def __init__(self, model, loss_fn):
        self.model = model
        self.loss_fn = loss_fn
        
        phy_params = [model.k, model.c, model.M0]
        net_params = [p for n, p in model.named_parameters() if n not in ['k', 'c', 'M0']]
        
        self.optimizer = torch.optim.Adam([
            {'params': net_params, 'lr': 0.01}, 
            {'params': phy_params, 'lr': 0.2} 
        ])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.8)

    def train(self, train_data, epochs=2000):
        omega = train_data['omega']
        target = {'theta': train_data['theta'], 'phi': train_data['phi']}
        
        print(f"--- PINN 开始训练 (共 {epochs} 轮) ---")
        for epoch in range(1, epochs + 1):
            self.model.train()
            self.optimizer.zero_grad()
            
            pred = self.model(omega)
            
            # 延长 Pretrain 时间
            current_phase = "pretrain" if epoch < 800 else "physics"
            if current_phase == "pretrain":
                self.model.k.requires_grad = False
                self.model.c.requires_grad = False
                self.model.M0.requires_grad = False
            else:
                self.model.k.requires_grad = True
                self.model.c.requires_grad = True
                self.model.M0.requires_grad = True

            loss = self.loss_fn(self.model, omega, pred, target, phase=current_phase)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if epoch % 1000 == 0 or epoch == epochs:
                with torch.no_grad():
                    phy_out = self.model.physic_model(omega)
                    J_val, c_val, k_val, M0_val = phy_out['params']
                    curr_peak = np.sqrt(k_val.item() / J_val)
                    print(f"Epoch {epoch:4d} | Phase: {current_phase} | Loss: {loss.item():.5f} | Phys Peak: {curr_peak:.3f}")

# ==========================================
# 第二部分：传统物理公式拟合 (基于 Scipy)
# ==========================================

class ScipyFitter:
    def __init__(self, omega, theta_rad, phi_rad):
        self.omega = omega
        self.theta_rad = theta_rad
        self.phi_rad = phi_rad
        self.popt_amp = None
        self.popt_phase = None

    def amplitude_func(self, omega, A0, omega0, beta):
        """受迫振动幅频公式: A = A0 / sqrt((w0^2 - w^2)^2 + 4*beta^2*w^2)"""
        return A0 / np.sqrt((omega0**2 - omega**2)**2 + 4 * beta**2 * omega**2)

    def phase_func(self, omega, omega0, beta):
        """受迫振动相频公式: tan(phi) = 2*beta*w / (w0^2 - w^2)"""
        return np.arctan2(2 * beta * omega, omega0**2 - omega**2)

    def fit(self):
        print("\n--- Scipy 传统曲线拟合 ---")
        # 初始猜测
        max_idx = np.argmax(self.theta_rad)
        w_max = self.omega[max_idx]
        A_max = self.theta_rad[max_idx]
        
        # 猜测参数: [A0, omega0, beta]
        # 注意: 此处 A0 对应 PINN 中的 M0/J
        p0 = [A_max * (w_max**2) * 0.1, w_max, 0.05] 
        bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
        
        try:
            # 拟合幅频曲线
            self.popt_amp, _ = curve_fit(
                self.amplitude_func, self.omega, self.theta_rad, 
                p0=p0, bounds=bounds, maxfev=10000
            )
            
            A0_fit, w0_fit, beta_fit = self.popt_amp
            print(f"拟合成功:")
            print(f"  固有频率 omega0 = {w0_fit:.4f} rad/s")
            print(f"  阻尼系数 beta   = {beta_fit:.4f} s^-1")
            print(f"  阻尼比 zeta     = {beta_fit/w0_fit:.4f}")
            
            return {
                'omega0': w0_fit,
                'beta': beta_fit,
                'A0': A0_fit,
                'zeta': beta_fit/w0_fit
            }
        except Exception as e:
            print(f"Scipy 拟合失败: {e}")
            return None

# ==========================================
# 第三部分：数据加载与综合可视化
# ==========================================

class DataLoader():
    def __init__(self, filepath):
        self.filepath = filepath
        
    def process_data(self):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"找不到文件: {self.filepath}")
            
        df = pd.read_excel(self.filepath)
        # 确保按 T 排序 (omega 升序)
        df = df.sort_values(by='T', ascending=False)
        
        T = df['T'].values
        theta_deg = df['theta'].values
        phi_deg = df['phi'].values
        
        # 转换单位
        omega = 2 * np.pi / T
        theta_rad = np.radians(theta_deg)
        # 假设数据中的相位是角度制
        phi_rad = np.radians(phi_deg) 
        
        # 估算数据中的共振峰位置
        max_idx = np.argmax(theta_rad)
        omega_peak = omega[max_idx]
        print(f"数据加载完成: 共 {len(T)} 个点, 峰值频率 ≈ {omega_peak:.3f} rad/s")

        # 返回 Tensor 格式供 PINN 使用，Numpy 格式供 Scipy 使用
        return {
            'omega_t': torch.FloatTensor(omega).reshape(-1, 1),
            'theta_t': torch.FloatTensor(theta_rad).reshape(-1, 1),
            'phi_t': torch.FloatTensor(phi_rad).reshape(-1, 1),
            'omega_peak_est': omega_peak,
            'omega_np': omega,
            'theta_np': theta_rad,
            'phi_np': phi_rad,
            'theta_deg_np': theta_deg,
            'phi_deg_np': phi_deg
        }

def visualize_comparison(model, train_data, scipy_params):
    # 1. 准备数据
    omega_data = train_data['omega_np']
    theta_data_deg = train_data['theta_deg_np']
    # 将实验相位处理为负值 (滞后)
    phi_data_deg = -np.abs(train_data['phi_deg_np']) 
    
    # 获取 PINN 的 omega0 用于归一化
    model.eval()
    with torch.no_grad():
        phy_out = model.physic_model(train_data['omega_t'])
        omega0_pinn = phy_out['omega0'][0].item()
        delta_pinn = phy_out['delta'][0].item()
        
    # 生成密集网格
    omega_dense_np = np.linspace(omega_data.min()*0.8, omega_data.max()*1.2, 2000)
    omega_dense_t = torch.FloatTensor(omega_dense_np).reshape(-1, 1)
    
    # 2. 计算各模型的预测值
    
    # --- PINN 预测 ---
    with torch.no_grad():
        pinn_pred = model(omega_dense_t)
        theta_pinn_deg = np.degrees(pinn_pred['theta'].numpy().flatten())
        # PINN 输出通常是 [0, pi]，转为负角度 [0, -180]
        phi_pinn_deg = -np.degrees(pinn_pred['phi'].numpy().flatten())

    # --- Scipy 理论预测 ---
    scipy_fitter = ScipyFitter(None, None, None) # 仅用于调用函数
    if scipy_params:
        theta_scipy_rad = scipy_fitter.amplitude_func(omega_dense_np, scipy_params['A0'], scipy_params['omega0'], scipy_params['beta'])
        phi_scipy_rad = scipy_fitter.phase_func(omega_dense_np, scipy_params['omega0'], scipy_params['beta'])
        
        theta_scipy_deg = np.degrees(theta_scipy_rad)
        phi_scipy_deg = -np.degrees(phi_scipy_rad) # 理论公式 arctan2 通常在共振时为 pi/2, 取负为 -90
        
        omega0_scipy = scipy_params['omega0']
        zeta_scipy = scipy_params['zeta']
    else:
        theta_scipy_deg = np.zeros_like(omega_dense_np)
        phi_scipy_deg = np.zeros_like(omega_dense_np)
        omega0_scipy = 0
        zeta_scipy = 0

    # 3. 绘图
    # 统一使用 PINN 的 omega0 进行 X 轴归一化，方便对齐
    lambda_data = omega_data / omega0_pinn
    lambda_plot = omega_dense_np / omega0_pinn
    
    plt.figure(figsize=(14, 6))

    # === 幅频特性 ===
    plt.subplot(1, 2, 1)
    # 实验点
    plt.scatter(lambda_data, theta_data_deg, color='black', label='Exp Data', s=40, zorder=5, marker='x')
    # PINN 曲线
    plt.plot(lambda_plot, theta_pinn_deg, 'b-', label=f'PINN\n$\omega_0$={omega0_pinn:.2f}, $\zeta$={delta_pinn:.3f}', linewidth=2.5, alpha=0.8)
    # Scipy 曲线
    if scipy_params:
        plt.plot(lambda_plot, theta_scipy_deg, 'r--', label=f'Scipy\n$\omega_0$={omega0_scipy:.2f}, $\zeta$={zeta_scipy:.3f}', linewidth=2)
    
    plt.title('Amplitude-Frequency Response')
    plt.xlabel(r'Frequency Ratio $\lambda = \omega / \omega_{0,PINN}$')
    plt.ylabel(r'Amplitude $\theta$ (deg)')
    plt.axvline(1.0, color='gray', linestyle=':', alpha=0.5)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # === 相频特性 ===
    plt.subplot(1, 2, 2)
    # 实验点
    plt.scatter(lambda_data, phi_data_deg, color='black', label='Exp Data', s=40, zorder=5, marker='x')
    # PINN 曲线
    plt.plot(lambda_plot, phi_pinn_deg, 'b-', label='PINN Prediction', linewidth=2.5, alpha=0.8)
    # Scipy 曲线
    if scipy_params:
        plt.plot(lambda_plot, phi_scipy_deg, 'r--', label='Theoretical Fit', linewidth=2)
    
    plt.title('Phase-Frequency Response')
    plt.xlabel(r'Frequency Ratio $\lambda = \omega / \omega_{0,PINN}$')
    plt.ylabel(r'Phase Lag $\phi$ (deg)')
    plt.axvline(1.0, color='gray', linestyle=':', alpha=0.5)
    # 限制 Y 轴范围在合理区间 (例如 10 到 -190)
    plt.ylim(-200, 20)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def calculate_delta_multi_level(train_data):
    """ 多截面带宽法计算实验数据的阻尼比 """
    omega = train_data['omega_np']
    theta = train_data['theta_np'] # 使用弧度
    
    # 样条插值提高分辨率
    interp_func = interp1d(omega, theta, kind='cubic', bounds_error=False, fill_value=0)
    omega_dense = np.linspace(omega.min(), omega.max(), 10000)
    theta_dense = interp_func(omega_dense)
    
    idx_max = np.argmax(theta_dense)
    theta_max = theta_dense[idx_max]
    omega_n = omega_dense[idx_max]
    
    print(f"\n====== 多截面带宽法 (数据分析) ======")
    deltas = []
    ratios = [0.5, 0.6, 0.7071, 0.8, 0.85, 0.9] 
    
    print(f"{'Ratio':<10} | {'Delta':<10}")
    print("-" * 25)
    
    for R in ratios:
        target_val = theta_max * R
        # 寻找交点
        left_region = theta_dense[:idx_max]
        right_region = theta_dense[idx_max:]
        
        if len(left_region) > 0 and len(right_region) > 0:
            w1 = omega_dense[np.argmin(np.abs(left_region - target_val))]
            w2 = omega_dense[idx_max + np.argmin(np.abs(right_region - target_val))]
            bandwidth = w2 - w1
            
            if bandwidth > 0:
                factor = np.sqrt(1/(R**2) - 1)
                if factor > 1e-6:
                    delta_calc = bandwidth / (2 * omega_n * factor)
                    deltas.append(delta_calc)
                    print(f"{R:<10.4f} | {delta_calc:.4f}")

    mean_delta = np.mean(deltas) if deltas else 0
    print("-" * 25)
    print(f"统计平均阻尼比 Delta: {mean_delta:.4f}")
    return mean_delta

# ==========================================
# 主程序入口
# ==========================================

if __name__ == "__main__":
    try:
        # 1. 准备数据
        dataLoader = DataLoader('ExperimentData.xlsx')
        data = dataLoader.process_data()
        
        # 2. 运行 Scipy 传统拟合
        fitter = ScipyFitter(data['omega_np'], data['theta_np'], data['phi_np'])
        scipy_params = fitter.fit()
        
        # 3. 运行 PINN 训练
        max_theta = data['theta_t'].max().item()
        model = ForceViberationModel(max_theta, data['omega_peak_est'])
        loss_fn = PINNLoss(data['omega_peak_est'])
        
        # 为了演示快速效果，Epoch设为5000，需要更精准可设为15000
        trainer = PINNTrainer(model, loss_fn)
        # 传入包含Tensor的字典
        train_input = {
            'omega': data['omega_t'], 
            'theta': data['theta_t'], 
            'phi': data['phi_t'], 
            'omega_peak_est': data['omega_peak_est']
        }
        trainer.train(train_input, epochs=10000)
        
        # 4. 综合可视化与对比
        visualize_comparison(model, data, scipy_params)
        
        # 5. 额外的数据分析
        calculate_delta_multi_level(data)
        
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()