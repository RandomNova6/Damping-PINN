import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class DataLoader():
    def __init__(self, data_string):
        # 从字符串创建数据
        # 从字符串创建数据，跳过标题行
        data_lines = data_string.strip().split('\n')
        data = []
        for line in data_lines[1:]:  # 跳过第一行标题
            parts = line.split()
            if len(parts) >= 3:
                # 注意：数据顺序是 theta, T, phi
                data.append([float(parts[0]), float(parts[1]), float(parts[2])])
        
        self.raw_data = pd.DataFrame(data, columns=['theta_deg', 'T', 'phi_deg'])
        
    def processData(self):
        data = self.raw_data.sort_values(by='T', ascending=True)
        T = data['T'].values
        theta_deg = data['theta_deg'].values
        phi_deg = data['phi_deg'].values
        
        # 计算角频率 ω = 2π/T
        omega = 2 * np.pi / T
        
        # 将角度转换为弧度
        theta_rad = theta_deg * np.pi / 180.0
        phi_rad = phi_deg * np.pi / 180.0
        
        # 调整相位：将相位差转换为相对于驱动力的相位
        # 在受迫振动中，相位通常在0到π之间变化
        phi_rad_adjusted = np.where(phi_rad > np.pi, phi_rad - 2*np.pi, phi_rad)
        
        return omega, theta_rad, phi_rad_adjusted


class ForcedVibrationAnalyzer:
    def __init__(self, omega, amplitude, phase):
        """
        初始化分析器
        omega: 驱动力频率 (rad/s)
        amplitude: 振动幅值 (rad)
        phase: 相位差 (rad)
        """
        self.omega = omega
        self.amplitude = amplitude
        self.phase = phase
        
    def amplitude_fitting_function(self, omega, A0, omega0, beta):
        """
        受迫振动幅频特性理论公式
        A(ω) = A0 / sqrt((ω0^2 - ω^2)^2 + 4β^2ω^2)
        """
        return A0 / np.sqrt((omega0**2 - omega**2)**2 + 4 * beta**2 * omega**2)
    
    def phase_fitting_function(self, omega, omega0, beta):
        """
        受迫振动相频特性理论公式
        φ(ω) = arctan(2βω/(ω0^2 - ω^2))
        注意：需要处理象限问题，使用arctan2
        """
        return np.arctan2(2 * beta * omega, omega0**2 - omega**2)
    
    def fit_amplitude_curve(self, initial_guess=None):
        """
        非线性最小二乘法拟合幅频特性曲线
        """
        if initial_guess is None:
            # 初始猜测值: [A0, omega0, beta]
            max_amp = np.max(self.amplitude)
            omega_at_max = self.omega[np.argmax(self.amplitude)]
            initial_guess = [max_amp, omega_at_max, 0.1]
        
        # 设置参数边界
        bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
        
        # 执行拟合
        popt, pcov = curve_fit(
            self.amplitude_fitting_function, 
            self.omega, 
            self.amplitude, 
            p0=initial_guess,
            bounds=bounds,
            maxfev=10000
        )
        
        self.amp_params = popt
        self.amp_cov = pcov
        
        # 计算拟合曲线
        omega_fine = np.linspace(min(self.omega), max(self.omega), 1000)
        amp_fitted = self.amplitude_fitting_function(omega_fine, *popt)
        
        return omega_fine, amp_fitted, popt
    
    def analyze_resonance_region(self):
        """
        分析共振区域和半功率带
        """
        print("="*60)
        print("共振区域分析")
        print("="*60)
        
        # 找到最大振幅
        max_amp = np.max(self.amplitude)
        max_amp_idx = np.argmax(self.amplitude)
        omega_max = self.omega[max_amp_idx]
        
        print(f"最大振幅: {max_amp:.6f} rad")
        print(f"最大振幅对应的频率: {omega_max:.6f} rad/s")
        print(f"最大振幅对应的周期: {2*np.pi/omega_max:.6f} s")
        print(f"最大振幅位置索引: {max_amp_idx}")
        
        # 半功率点对应的振幅
        half_power_amp = max_amp / np.sqrt(2)
        print(f"\n半功率点振幅 (A_max/√2): {half_power_amp:.6f} rad")
        
        # 检查数据点
        print(f"\n所有数据点振幅:")
        for i, (w, a) in enumerate(zip(self.omega, self.amplitude)):
            print(f"  ω={w:.3f} rad/s, A={a:.6f} rad, 是否低于半功率: {a < half_power_amp}")
        
        # 寻找半功率点
        # 使用插值找到精确的半功率点
        try:
            # 创建插值函数
            amp_interp = interp1d(self.omega, self.amplitude, kind='cubic', bounds_error=False, fill_value='extrapolate')
            
            # 在更精细的网格上计算
            omega_fine = np.linspace(min(self.omega), max(self.omega), 1000)
            amp_fine = amp_interp(omega_fine)
            
            # 找到半功率点的位置
            half_power_idx = np.where(amp_fine < half_power_amp)[0]
            
            if len(half_power_idx) > 0:
                print(f"\n在插值曲线上找到 {len(half_power_idx)} 个低于半功率的点")
                
                # 找到左侧半功率点
                left_idx = half_power_idx[half_power_idx < np.argmax(amp_fine)]
                if len(left_idx) > 0:
                    left_idx = left_idx[-1]
                    omega_left = omega_fine[left_idx]
                    print(f"左侧半功率点: ω={omega_left:.6f} rad/s")
                else:
                    omega_left = None
                    print("左侧半功率点: 未找到")
                
                # 找到右侧半功率点
                right_idx = half_power_idx[half_power_idx > np.argmax(amp_fine)]
                if len(right_idx) > 0:
                    right_idx = right_idx[0]
                    omega_right = omega_fine[right_idx]
                    print(f"右侧半功率点: ω={omega_right:.6f} rad/s")
                else:
                    omega_right = None
                    print("右侧半功率点: 未找到")
                
                if omega_left is not None and omega_right is not None:
                    # 计算半功率带宽
                    delta_omega = omega_right - omega_left
                    
                    # 计算阻尼系数：β = Δω/2
                    beta_half = delta_omega / 2
                    
                    # 计算品质因数
                    Q_half = omega_max / delta_omega
                    
                    print(f"\n半功率带宽 Δω: {delta_omega:.6f} rad/s")
                    print(f"阻尼系数 β = Δω/2: {beta_half:.6f} rad/s")
                    print(f"品质因数 Q = ω_max/Δω: {Q_half:.6f}")
                    
                    return {
                        '最大振幅': max_amp,
                        '共振频率': omega_max,
                        '左侧半功率频率': omega_left,
                        '右侧半功率频率': omega_right,
                        '半功率带宽': delta_omega,
                        '阻尼系数': beta_half,
                        '品质因数': Q_half
                    }
                else:
                    print("\n警告：未找到完整的半功率点，可能原因：")
                    print("1. 数据点不足")
                    print("2. 系统阻尼太小")
                    print("3. 数据测量范围不够宽")
                    
                    # 尝试直接使用数据点
                    return self._estimate_from_data_points(max_amp, omega_max, half_power_amp)
            else:
                print("\n警告：插值曲线上没有找到低于半功率的点")
                return self._estimate_from_data_points(max_amp, omega_max, half_power_amp)
                
        except Exception as e:
            print(f"\n插值时出错: {e}")
            return self._estimate_from_data_points(max_amp, omega_max, half_power_amp)
    
    def _estimate_from_data_points(self, max_amp, omega_max, half_power_amp):
        """直接从数据点估计"""
        print("\n尝试直接从数据点估计...")
        
        # 寻找最接近半功率点的数据点
        left_candidates = []
        right_candidates = []
        
        for i, (w, a) in enumerate(zip(self.omega, self.amplitude)):
            if a < half_power_amp:
                if w < omega_max:
                    left_candidates.append((w, a))
                else:
                    right_candidates.append((w, a))
        
        print(f"左侧候选点: {len(left_candidates)} 个")
        print(f"右侧候选点: {len(right_candidates)} 个")
        
        if left_candidates and right_candidates:
            # 选择最接近半功率振幅的点
            left_w, left_a = max(left_candidates, key=lambda x: x[1])  # 左侧最大值
            right_w, right_a = min(right_candidates, key=lambda x: x[0])  # 右侧最小值
            
            print(f"选择的左侧点: ω={left_w:.6f}, A={left_a:.6f}")
            print(f"选择的右侧点: ω={right_w:.6f}, A={right_a:.6f}")
            
            delta_omega = right_w - left_w
            beta_est = delta_omega / 2
            Q_est = omega_max / delta_omega
            
            return {
                '最大振幅': max_amp,
                '共振频率': omega_max,
                '左侧频率': left_w,
                '右侧频率': right_w,
                '带宽估计': delta_omega,
                '阻尼系数估计': beta_est,
                '品质因数估计': Q_est,
                '备注': '基于最近数据点估计'
            }
        else:
            print("无法找到合适的数据点进行估计")
            return None
    
    def plot_amplitude_phase(self):
        """
        绘制振幅和相位随频率变化的曲线
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 幅频特性图
        ax1.plot(self.omega, self.amplitude, 'bo-', linewidth=2, markersize=8, label='实验数据')
        max_amp = np.max(self.amplitude)
        max_amp_idx = np.argmax(self.amplitude)
        omega_max = self.omega[max_amp_idx]
        
        # 标记共振点
        ax1.plot(omega_max, max_amp, 'r*', markersize=15, label=f'共振点 (ω={omega_max:.3f})')
        
        # 绘制半功率线
        half_power_amp = max_amp / np.sqrt(2)
        ax1.axhline(y=half_power_amp, color='gray', linestyle='--', alpha=0.7)
        ax1.text(min(self.omega), half_power_amp*1.05, 
                f'A_max/√2 = {half_power_amp:.4f}', fontsize=10)
        
        ax1.set_xlabel('角频率 ω (rad/s)', fontsize=12)
        ax1.set_ylabel('振幅 A (rad)', fontsize=12)
        ax1.set_title('受迫振动幅频特性曲线', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 相频特性图
        ax2.plot(self.omega, self.phase, 'go-', linewidth=2, markersize=8, label='实验数据')
        
        # 标记共振点附近的相位（理论上在共振点相位=π/2）
        resonance_phase = self.phase[max_amp_idx] if max_amp_idx < len(self.phase) else None
        if resonance_phase is not None:
            ax2.plot(omega_max, resonance_phase, 'r*', markersize=15, 
                    label=f'共振点相位={resonance_phase:.3f}')
        
        ax2.set_xlabel('角频率 ω (rad/s)', fontsize=12)
        ax2.set_ylabel('相位 φ (rad)', fontsize=12)
        ax2.set_title('受迫振动相频特性曲线', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_detailed_analysis(self):
        """
        绘制详细分析图
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 左图：振幅随频率变化，展示半功率带
        ax1.plot(self.omega, self.amplitude, 'bo-', linewidth=2, markersize=8)
        
        max_amp = np.max(self.amplitude)
        max_amp_idx = np.argmax(self.amplitude)
        omega_max = self.omega[max_amp_idx]
        
        # 标记共振点
        ax1.plot(omega_max, max_amp, 'r*', markersize=15)
        
        # 半功率线
        half_power_amp = max_amp / np.sqrt(2)
        ax1.axhline(y=half_power_amp, color='gray', linestyle='--', alpha=0.7)
        
        # 尝试找到半功率点
        try:
            amp_interp = interp1d(self.omega, self.amplitude, kind='cubic')
            omega_fine = np.linspace(min(self.omega), max(self.omega), 1000)
            amp_fine = amp_interp(omega_fine)
            
            # 找到半功率点
            half_idx = np.where(amp_fine < half_power_amp)[0]
            left_idx = half_idx[half_idx < np.argmax(amp_fine)]
            right_idx = half_idx[half_idx > np.argmax(amp_fine)]
            
            if len(left_idx) > 0 and len(right_idx) > 0:
                left_idx = left_idx[-1]
                right_idx = right_idx[0]
                
                omega_left = omega_fine[left_idx]
                omega_right = omega_fine[right_idx]
                
                # 绘制半功率点
                ax1.plot(omega_left, half_power_amp, 'ms', markersize=10)
                ax1.plot(omega_right, half_power_amp, 'ms', markersize=10)
                
                # 绘制带宽线
                ax1.plot([omega_left, omega_right], 
                        [half_power_amp, half_power_amp], 'k-', linewidth=3)
                
                # 标注
                ax1.annotate('', xy=(omega_right, half_power_amp), 
                           xytext=(omega_left, half_power_amp),
                           arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
                ax1.text((omega_left+omega_right)/2, half_power_amp*1.1,
                       f'Δω = {omega_right-omega_left:.3f}',
                       ha='center', color='purple', fontsize=12, fontweight='bold')
        except:
            pass
        
        ax1.set_xlabel('角频率 ω (rad/s)', fontsize=12)
        ax1.set_ylabel('振幅 A (rad)', fontsize=12)
        ax1.set_title('幅频特性与半功率带', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # 右图：数据表格和分析结果
        ax2.axis('tight')
        ax2.axis('off')
        
        # 数据统计
        info_text = "数据统计:\n"
        info_text += f"数据点数: {len(self.omega)}\n"
        info_text += f"频率范围: {min(self.omega):.3f} - {max(self.omega):.3f} rad/s\n"
        info_text += f"周期范围: {2*np.pi/max(self.omega):.3f} - {2*np.pi/min(self.omega):.3f} s\n"
        info_text += f"\n振幅统计:\n"
        info_text += f"最大振幅: {max_amp:.6f} rad\n"
        info_text += f"最大振幅频率: {omega_max:.3f} rad/s\n"
        info_text += f"半功率振幅: {half_power_amp:.6f} rad\n"
        info_text += f"\n相位统计:\n"
        info_text += f"相位范围: {min(self.phase):.3f} - {max(self.phase):.3f} rad\n"
        if max_amp_idx < len(self.phase):
            info_text += f"共振点相位: {self.phase[max_amp_idx]:.3f} rad\n"
        
        ax2.text(0.1, 0.5, info_text, fontsize=11, 
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.show()


def main():
    # 您的数据
    data_string = """theta	T	phi
37.5	1.643	164
45.5	1.658	161
52.5	1.668	157
61.5	1.678	153
69.5	1.686	149
77.5	1.692	144
86.5	1.698	139
96.5	1.704	134
105	1.709	128
115.5	1.715	119
126	1.722	107
131	1.72	95
131	1.731	90
131	1.73	92
131	1.732	88
131	1.733	86
127.5	1.737	73
116.5	1.746	64
105.5	1.753	55
96.5	1.758	49"""
    
    print("="*60)
    print("受迫振动数据分析")
    print("="*60)
    
    # 1. 加载数据
    data_loader = DataLoader(data_string)
    omega, amplitude, phase = data_loader.processData()
    
    print("\n原始数据转换结果:")
    print(f"角频率 ω (rad/s): {omega}")
    print(f"振幅 A (rad): {amplitude}")
    print(f"相位 φ (rad): {phase}")
    
    # 2. 创建分析器
    analyzer = ForcedVibrationAnalyzer(omega, amplitude, phase)
    
    # 3. 分析共振区域
    print("\n" + "="*60)
    print("开始分析共振区域和半功率带")
    print("="*60)
    
    results = analyzer.analyze_resonance_region()
    
    if results:
        print("\n" + "="*60)
        print("分析结果总结")
        print("="*60)
        for key, value in results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")
    
    # 4. 绘制图形
    print("\n" + "="*60)
    print("绘制分析图形")
    print("="*60)
    
    analyzer.plot_amplitude_phase()
    analyzer.plot_detailed_analysis()
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()