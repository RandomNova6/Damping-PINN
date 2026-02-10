import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw_data = None
        
    def load_data(self):
        """
        读取Excel文件
        Excel应包含表头：theta, T, phi
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"找不到文件: {self.file_path}")
            
        try:
            # 读取 Excel 文件
            df = pd.read_excel(self.file_path)
            
            # 检查列名，兼容一些常见的变体
            required_cols = ['theta', 'T', 'phi']
            df.columns = [c.strip() for c in df.columns] # 去除列名空格
            
            # 简单的列名映射检查
            if not all(col in df.columns for col in required_cols):
                print(f"警告: Excel列名应包含 {required_cols}，实际列名为 {df.columns.tolist()}")
                # 尝试按顺序读取前三列
                df = df.iloc[:, :3]
                df.columns = ['theta', 'T', 'phi']
            
            self.raw_data = df
            print(f"成功读取数据，共 {len(df)} 行。")
            
        except Exception as e:
            raise Exception(f"读取Excel失败: {e}")

    def process_data(self):
        if self.raw_data is None:
            self.load_data()
            
        data = self.raw_data.sort_values(by='T', ascending=False) # T 越大，omega 越小
        
        T = data['T'].values
        theta_deg = data['theta'].values  # 幅值 (度)
        phi_deg = data['phi'].values      # 相位 (度)
        
        # 1. 计算角频率 omega = 2pi / T
        omega = 2 * np.pi / T
        
        # 2. 幅值转换为弧度 (用于拟合计算)
        theta_rad = np.radians(theta_deg)
        
        # 3. 相位处理
        # 原始数据通常是直接读数，需转换为弧度
        # 注意：受迫振动相位差通常定义为滞后，范围 [0, 180] 或 [0, -180]
        phi_rad = np.radians(phi_deg)
        
        return omega, theta_rad, phi_rad, theta_deg, phi_deg


class ForcedVibrationAnalyzer:
    def __init__(self, omega, amplitude_rad, phase_rad, amplitude_deg, phase_deg):
        self.omega = omega
        self.amp_rad = amplitude_rad
        self.phase_rad = phase_rad
        self.amp_deg = amplitude_deg
        self.phase_deg = phase_deg
        
        # 存储拟合参数
        self.popt_amp = None
        self.omega0_fit = None
        self.beta_fit = None
        
    def amplitude_func(self, omega, A0, omega0, beta):
        """受迫振动幅频公式 (理论模型)"""
        return A0 / np.sqrt((omega0**2 - omega**2)**2 + 4 * beta**2 * omega**2)
    
    def phase_func(self, omega, omega0, beta):
        """受迫振动相频公式 (理论模型)"""
        # arctan2 返回值范围 (-pi, pi]
        # 物理上相位滞后 phi = arctan(2*beta*omega / (omega0^2 - omega^2))
        return np.arctan2(2 * beta * omega, omega0**2 - omega**2)

    def fit_parameters(self):
        """拟合物理参数 omega0, beta"""
        # 初始猜测
        max_idx = np.argmax(self.amp_rad)
        w_max = self.omega[max_idx]
        A_max = self.amp_rad[max_idx]
        
        # 猜测: A0 约等于 A_max * 2*beta*w0 (近似), omega0 约等于 w_max
        p0 = [A_max/10, w_max, 0.05] 
        bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
        
        try:
            self.popt_amp, _ = curve_fit(
                self.amplitude_func, 
                self.omega, 
                self.amp_rad, 
                p0=p0, 
                bounds=bounds,
                maxfev=10000
            )
            self.omega0_fit = self.popt_amp[1]
            self.beta_fit = self.popt_amp[2]
            
            print(f"拟合成功:")
            print(f"  固有频率 omega0 = {self.omega0_fit:.4f} rad/s")
            print(f"  阻尼系数 beta   = {self.beta_fit:.4f} s^-1")
            print(f"  阻尼比 zeta     = {self.beta_fit/self.omega0_fit:.4f}")
            
        except Exception as e:
            print(f"拟合失败: {e}")
            # 如果拟合失败，使用最大值作为估算
            self.omega0_fit = w_max
            self.beta_fit = 0.1
            self.popt_amp = [A_max, w_max, 0.1]

    def visualize_like_target(self):
        """
        仿照目标格式绘图
        """
        if self.omega0_fit is None:
            self.fit_parameters()
            
        omega0 = self.omega0_fit
        beta = self.beta_fit
        A0 = self.popt_amp[0]
        
        # 1. 生成密集网格用于画平滑曲线
        omega_min = np.min(self.omega) * 0.8
        omega_max = np.max(self.omega) * 1.2
        omega_dense = np.linspace(omega_min, omega_max, 2000)
        
        # 2. 计算理论曲线
        # 幅值 (计算出弧度 -> 转角度)
        amp_theory_rad = self.amplitude_func(omega_dense, A0, omega0, beta)
        amp_theory_deg = np.degrees(amp_theory_rad)
        
        # 相位 (计算出弧度 -> 转角度)
        # 注意：理论公式通常给出正值的相位滞后，或者数学上的相角。
        # 目标图中相位为负值 (0 到 -180)，表示滞后。
        # 标准物理公式 arctan2(y, x) 在共振时为 pi/2。
        # 我们将其取负并转换为度数，以匹配 "Phase Difference (deg)" 为负的习惯
        phase_theory_rad = self.phase_func(omega_dense, omega0, beta)
        phase_theory_deg = -np.degrees(phase_theory_rad) 
        
        # 3. 处理实验数据的坐标
        # 横坐标：归一化频率 lambda
        lambda_data = self.omega / omega0
        lambda_plot = omega_dense / omega0
        
        # 纵坐标：
        # 幅值直接使用 self.amp_deg
        # 相位需要根据实验数据的记录方式调整。
        # 假设输入数据 phi 是正值 (0~180)，我们在图中显示为负值以表示滞后
        # 如果原始数据已经是负的，则不需要加负号。这里假设原始数据记录的是滞后量的绝对值。
        phase_data_plot = -np.abs(self.phase_deg) 

        # 4. 开始绘图
        plt.figure(figsize=(12, 5))
        
        # === 子图1：幅频特性 ===
        plt.subplot(1, 2, 1)
        # 实验数据点
        plt.scatter(lambda_data, self.amp_deg, color='black', label='Exp Data', s=30, zorder=3, marker='x')
        # 理论曲线
        plt.plot(lambda_plot, amp_theory_deg, 'r--', label='Physics Theory', linewidth=2)
        
        plt.title('Amplitude-Frequency Response')
        plt.xlabel(r'Frequency Ratio $\lambda = \omega / \omega_0$')
        plt.ylabel(r'Amplitude $\theta$ (deg)')
        plt.axvline(1.0, color='gray', linestyle=':', alpha=0.5, label='Resonance')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # === 子图2：相频特性 ===
        plt.subplot(1, 2, 2)
        # 实验数据点
        plt.scatter(lambda_data, phase_data_plot, color='black', label='Exp Data', s=30, zorder=3, marker='x')
        # 理论曲线
        plt.plot(lambda_plot, phase_theory_deg, 'r--', label='Physics Theory', linewidth=2)
        
        plt.title('Phase-Frequency Response')
        plt.xlabel(r'Frequency Ratio $\lambda = \omega / \omega_0$')
        plt.ylabel(r'Phase Difference $\phi$ (deg)')
        plt.axvline(1.0, color='gray', linestyle=':', alpha=0.5)
        # 设置Y轴范围，通常相位在 0 到 -180 之间
        plt.ylim(-190, 10)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()

# ==========================================
# 辅助函数：创建一个测试用的 Excel 文件
# (如果你已有文件，可以忽略此部分)
# ==========================================
def create_dummy_excel(filename="vibration_data.xlsx"):
    data = {
        'theta': [37.5, 45.5, 52.5, 61.5, 69.5, 77.5, 86.5, 96.5, 105, 115.5, 
                  126, 131, 131, 131, 131, 131, 127.5, 116.5, 105.5, 96.5],
        'T': [1.643, 1.658, 1.668, 1.678, 1.686, 1.692, 1.698, 1.704, 1.709, 1.715, 
              1.722, 1.72, 1.731, 1.73, 1.732, 1.733, 1.737, 1.746, 1.753, 1.758],
        'phi': [164, 161, 157, 153, 149, 144, 139, 134, 128, 119, 
                107, 95, 90, 92, 88, 86, 73, 64, 55, 49]
    }
    df = pd.DataFrame(data)
    # 注意：原始数据的phi随着omega增加(T减小)而减小，
    # 这可能意味着记录的是相位"导前"或者与标准定义的角度互补。
    # 标准受迫振动：omega增加 -> 相位滞后增加 (0 -> 90 -> 180)。
    # 这里的 phi 看起来像是 180 - lag。为了演示效果，代码会自动拟合。
    
    df.to_excel(filename, index=False)
    print(f"已创建测试文件: {filename}")

# ==========================================
# 主程序
# ==========================================
def main():
    # 文件路径
    excel_file = 'ExperimentData.xlsx'
    
    print("="*60)
    print("受迫振动数据分析 (Excel模式)")
    print("="*60)
    
    try:
        # 2. 加载数据
        loader = DataLoader(excel_file)
        loader.load_data()
        omega, amp_rad, phi_rad, amp_deg, phi_deg = loader.process_data()
        
        # 3. 初始化分析器
        analyzer = ForcedVibrationAnalyzer(omega, amp_rad, phi_rad, amp_deg, phi_deg)
        
        # 4. 执行拟合与绘图
        analyzer.fit_parameters()
        analyzer.visualize_like_target()
        
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()