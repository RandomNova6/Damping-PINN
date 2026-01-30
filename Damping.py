import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ForceViberationModel(nn.Module):
    def __init__(self,max_theta_data,omega_peak_data):
        super().__init__()
        
        # --- 神经网络 (保持不变) ---
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2) 
        )
        
        self.J_fixed = 0.2
        real_k = self.J_fixed * (omega_peak_data ** 2)
        
        # 初始 c 设为临界阻尼的 5% (保证有明显的尖峰)
        real_c = 0.02 * (2 * np.sqrt(self.J_fixed * real_k))
        
        # --- 关键：根据物理公式推算 M0 ---
        # Theta_max = M0 / (c * omega_peak) -> M0 = Theta_max * c * omega_peak
        real_M0 = max_theta_data * real_c * omega_peak_data

        print(f"设定 k = {real_k:.5f}, c = {real_c:.5f}, M0 = {real_M0:.5f}")
        
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
        
        # --- 相位公式 ---
        # 标准物理相位 (0 -> pi)
        phi_phy = torch.atan2(c * omega_val, k - J * omega_sq)
        
        # 计算辅助指标
        omega0 = torch.sqrt(k / J)
        delta = c / (2 * torch.sqrt(J * k))
        
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
        phi = output[:, 1] # 相位让它自由一点，不要强加 softplus
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
        
        # 频率对齐 (你之前的策略，非常好)
        loss_omega0 = self.mse(phy_out['omega0'], self.omega_target.to(omega.device))
        
        # 3. 阻尼惩罚 (防止阻尼过大导致曲线变平)
        # 阻尼比 delta = c / (2*sqrt(J*k))，我们希望它保持在欠阻尼状态 (delta < 1)
        reg_loss = torch.relu(phy_out['delta'] - 0.5) * 10.0 

        return data_total + 10.0 * (loss_theta_phy + loss_phi_phy) + 100.0 * loss_omega0 + 1.0*reg_loss
# --------------------------
# 2. 修正后的训练器 (差分学习率)
# --------------------------
class PINNTrainer():
    def __init__(self, model, loss_fn):
        self.model = model
        self.loss_fn = loss_fn
        
        # 进一步降低学习率，防止参数乱飞
        phy_params = [model.k, model.c, model.M0]
        net_params = [p for n, p in model.named_parameters() if n not in ['k', 'c', 'M0']]
        
        self.optimizer = torch.optim.Adam([
            {'params': net_params, 'lr': 0.01}, 
            {'params': phy_params, 'lr': 0.2}  # 把物理参数的学习率压得极低
        ])
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.8)

    def train(self, train_data, epochs=2000):
        omega = train_data['omega']
        target = {'theta': train_data['theta'], 'phi': train_data['phi']}
        
        print(f"--- 开始训练 (共 {epochs} 轮) ---")
        
        for epoch in range(1, epochs + 1):
            self.model.train()
            self.optimizer.zero_grad()
            
            pred = self.model(omega)
            
            # 延长 Pretrain 时间，确保 NN 已经完美拟合了数据
            current_phase = "pretrain" if epoch < 600 else "physics"
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

            if epoch % 100 == 0:
                with torch.no_grad():
                    phy_out = self.model.physic_model(omega)
                    J_val, c_val, k_val, M0_val = phy_out['params']
                    # 计算当前物理模型的共振频率
                    curr_peak = np.sqrt(k_val.item() / J_val)
                    
                    print(f"Epoch {epoch:4d} | Phase: {current_phase:8s} | Loss: {loss.item():.5f}")
                    print(f"    --> Phys Peak: {curr_peak:.3f} rad/s (Data Peak: {train_data['omega_peak_est']:.3f})")
                    print(f"    --> k: {k_val.item():.4f}, c: {c_val.item():.4f}")

        model.eval()
        with torch.no_grad():
            final_phy = model.physic_model(train_data['omega'])
            J, c, k, M0 = final_phy['params']
            delta = final_phy['delta']
            
            print("\n--- 最终物理参数定型 ---")
            print(f"转动惯量 J: {J:.6f}")
            print(f"弹性系数 k: {k.item():.6f}")
            print(f"阻尼系数 c: {c.item():.6f}")
            print(f"激励力矩 M0: {M0.item():.6f}")
            print(f"阻尼比 Delta: {delta.item():.6f}")

#数据格式：theta, T, phi, delta
class DataLoader():
    def __init__(self, filepath):
        self.raw_data = pd.read_excel(filepath)
        
    def process_data(self):
        # 确保按 T 排序，这样画图时连线才不会乱
        self.raw_data = self.raw_data.sort_values(by='T', ascending=False)
        
        # 提取数据
        T = self.raw_data['T'].values
        theta_deg = self.raw_data['theta'].values
        phi_deg = self.raw_data['phi'].values
        
        # 转换单位
        omega = 2 * np.pi / T
        theta_rad = theta_deg * np.pi / 180.0
        phi_rad = phi_deg * np.pi / 180.0 
        
        # 估算数据中的共振峰位置 (用于初始化)
        max_idx = np.argmax(theta_rad)
        omega_peak = omega[max_idx]
        print(f"检测到数据峰值位于 ω ≈ {omega_peak:.3f} rad/s")

        return {
            'omega': torch.FloatTensor(omega).reshape(-1, 1),
            'theta': torch.FloatTensor(theta_rad).reshape(-1, 1),
            'phi': torch.FloatTensor(phi_rad).reshape(-1, 1),
            'omega_peak_est': omega_peak
        }
    
def calculate_delta_from_curve(omega_dense, theta_nn):
    # 1. 找峰值
    idx_max = np.argmax(theta_nn)
    theta_max = theta_nn[idx_max]
    omega_0 = omega_dense[idx_max]
    
    # 2. 找 0.707 倍高度的频率点
    target_theta = theta_max / np.sqrt(2)
    # 找到峰值左侧和右侧最接近 target_theta 的点
    left_side = theta_nn[:idx_max]
    right_side = theta_nn[idx_max:]
    
    omega_1 = omega_dense[np.argmin(np.abs(left_side - target_theta))]
    omega_2 = omega_dense[idx_max + np.argmin(np.abs(right_side - target_theta))]
    
    # 3. 计算阻尼比
    delta_img = (omega_2 - omega_1) / (2 * omega_0)
    return delta_img, omega_1, omega_2
    
def visualize_results(model, train_data):
    omega_data = train_data['omega'].detach().cpu().numpy().flatten()
    theta_data = train_data['theta'].detach().cpu().numpy().flatten()
    phi_data = train_data['phi'].detach().cpu().numpy().flatten()

    omega_min = np.min(omega_data) * 0.9
    omega_max = np.max(omega_data) * 1.1
    omega_dense = torch.linspace(omega_min, omega_max, 500).reshape(-1, 1)

    model.eval()
    with torch.no_grad():
        nn_pred = model(omega_dense)
        phy_pred = model.physic_model(omega_dense)

    omega_plot = omega_dense.numpy().flatten()
    theta_nn = nn_pred['theta'].numpy().flatten()
    phi_nn = nn_pred['phi'].numpy().flatten()
    theta_phy = phy_pred['theta'].numpy().flatten()
    phi_phy = phy_pred['phi'].numpy().flatten()
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(omega_data, theta_data, color='black', label='Exp Data', s=30, zorder=3, marker='x')
    plt.plot(omega_plot, theta_nn, 'b-', label='NN Prediction', linewidth=2, alpha=0.8)
    plt.plot(omega_plot, theta_phy, 'r--', label='Physics Theory', linewidth=2)
    
    plt.title('Amplitude-Frequency Response')
    plt.xlabel(r'Angular Frequency $\omega$ (rad/s)')
    plt.ylabel(r'Amplitude $\theta$ (rad)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.subplot(1, 2, 2)

    plt.scatter(omega_data, phi_data, color='black', label='Exp Data', s=30, zorder=3, marker='x')
    plt.plot(omega_plot, phi_nn, 'b-', label='NN Prediction', linewidth=2, alpha=0.8)
    plt.plot(omega_plot, phi_phy, 'r--', label='Physics Theory', linewidth=2)

    delta_img, w1, w2 = calculate_delta_from_curve(omega_plot, theta_nn)
    print(f"--- 图像提取分析 ---")
    print(f"基于拟合曲线计算的阻尼比 Delta: {delta_img:.4f}")
    
    plt.title('Phase-Frequency Response')
    plt.xlabel(r'Angular Frequency $\omega$ (rad/s)')
    plt.ylabel(r'Phase $\phi$ (rad)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dataLoader = DataLoader('ExperimentData.xlsx')
    train_data = dataLoader.process_data()
    
    # 获取数据中的最大振幅
    max_theta = train_data['theta'].max().item()

    # 初始化模型时传入 max_theta
    model = ForceViberationModel(train_data['omega_peak_est'], max_theta)
    loss_fn = PINNLoss(train_data['omega_peak_est'])

    trainer = PINNTrainer(model, loss_fn)
    trainer.train(train_data, epochs=25000)
    visualize_results(model, train_data)
