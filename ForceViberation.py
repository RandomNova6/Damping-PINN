import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim

class ForceViberationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(
            nn.Linear(1,32),nn.Tanh(),
            nn.Linear(32,32),nn.Tanh(),
            nn.Linear(32,32),nn.Tanh(),
            nn.Linear(32,2)
        )

    def forward(self,in_feature:torch.Tensor)->torch.Tensor:
        return self.model(in_feature)

class PINNLoss(nn.Module):
    def __init__(self, zeta_init=0.1):
        super().__init__()
        self.zeta = nn.Parameter(torch.tensor([zeta_init], dtype=torch.float32))
        self.A0 = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))

    def forward(self, eta, pred_theta, pred_phi, true_theta, true_phi):
        #数据损失
        loss_data_theta = torch.mean((pred_theta - true_theta)**2)
        loss_data_phi = torch.mean((pred_phi - true_phi)**2)
        
        #幅频物理残差
        physics_res_phi = torch.sin(pred_phi) * (1 - eta**2)-2 * self.zeta * eta * torch.cos(pred_phi)
        loss_physics_phi = torch.mean(physics_res_phi**2)

        #相频物理残差
        factor = (1 - eta**2)**2 + (2 * self.zeta * eta)**2
        physics_res_theta = pred_theta**2 * factor - self.A0**2
        loss_physics_theta = torch.mean(physics_res_theta**2)

        total_loss = (loss_data_theta + loss_data_phi) + \
                     0.1 * (loss_physics_phi + loss_physics_theta)
        
        return total_loss

#接收数据格式：['theta', 'T', 'phi']
#返回数据格式:['theta','eta(omega_rate)','phi']
class DataLoader:
    def __call__(self,filepath:str)->dict:
        data=self.read_data(filepath)
        processed_data=self.process_data(data)
        return processed_data

    def read_data(self,filepath:str)->pd.DataFrame:
        data=pd.read_excel(filepath)
        return data
    
    def process_data(self,data:pd.DataFrame)->dict:
        assert data.columns.equals(pd.Index(['theta', 'T', 'phi']))
        data = data.sort_values(by='T',ascending=False).reset_index(drop=True)

        theta=data['theta']/180*np.pi
        T=data['T']
        phi=data['phi']/180*np.pi
        
        omega=2*np.pi/T
        omega_peak=self.find_peak(theta,omega)
        eta=omega/omega_peak
        return {
            'theta':torch.tensor(theta.values, dtype=torch.float32).view(-1,1),
            'eta':torch.tensor(eta.values, dtype=torch.float32).view(-1,1),
            'phi':torch.tensor(phi.values, dtype=torch.float32).view(-1,1),
            'omega_peak':omega_peak
        }

    def find_peak(self,theta:pd.Series,omega:pd.Series)->int:
        max_idx = theta.idxmax()
        omega_peak = omega[max_idx]
        return omega_peak
    
def train_pinn(data, epochs=5000, lr=0.001):
    model = ForceViberationModel()
    criterion = PINNLoss(zeta_init=0.1) 

    optimizer = optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=lr)

    eta = data['eta']
    true_theta = data['theta']
    true_phi = data['phi']

    loss_history = []
    zeta_history = []

    print("开始训练 PINN...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        preds = model(eta)
        pred_theta = preds[:, 0:1]
        pred_phi = preds[:, 1:2]

        loss = criterion(eta, pred_theta, pred_phi, true_theta, true_phi)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            criterion.zeta.clamp_(min=1e-4)
            criterion.A0.clamp_(min=1e-4)

        loss_history.append(loss.item())
        zeta_history.append(criterion.zeta.item())

        if epoch % 500 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f} | zeta: {criterion.zeta.item():.4f} | A0: {criterion.A0.item():.4f}")

    return model,criterion,loss_history,zeta_history

def plot_pinn_results(data_dict,model, criterion):
    eta_test = torch.linspace(data_dict['eta'].min()*0.9, 
                              data_dict['eta'].max()*1.1, 200).view(-1, 1)
    
    model.eval()
    with torch.no_grad():
        preds = model(eta_test)
        pred_theta = preds[:, 0:1].numpy()
        pred_phi = preds[:, 1:2].numpy() * (180 / np.pi)

    exp_eta = data_dict['eta'].numpy()
    exp_theta = data_dict['theta'].numpy()
    exp_phi = data_dict['phi'].numpy() * (180 / np.pi)

    plt.figure(figsize=(12, 5))

    # 幅频特性图
    plt.subplot(1, 2, 1)
    plt.scatter(exp_eta, exp_theta, c='red', alpha=0.5, label='Experimental Data')
    plt.plot(eta_test.numpy(), pred_theta, 'b-', label='PINN Prediction')
    plt.axvline(x=1.0, color='gray', linestyle='--', label='Resonance Point')
    plt.title(f"Amplitude-Frequency (Found $\zeta$: {criterion.zeta.item():.3f})")
    plt.xlabel("$\omega / \omega_0$")
    plt.ylabel("Amplitude (rad)")
    plt.legend()

    # 相频特性图
    plt.subplot(1, 2, 2)
    plt.scatter(exp_eta, exp_phi, c='red', alpha=0.5, label='Experimental Data')
    plt.plot(eta_test.numpy(), pred_phi, 'b-', label='PINN Prediction')
    plt.axhline(y=-90, color='gray', linestyle='--')
    plt.title("Phase-Frequency")
    plt.xlabel("$\omega / \omega_0$")
    plt.ylabel("Phase Lag (degrees)")
    plt.legend()

    plt.tight_layout()
    plt.show()

'''
    x_epoches=np.linspace(1,epoches+1,epoches)

    plt.subplot(2, 2, 3)
    plt.plot(x_epoches, loss_history, 'b-', label='Loss')
    plt.yscale('log')
    plt.title("Loss History (Log Scale)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True, which="both", ls="-", alpha=0.5) # 增加网格线方便查看数量级
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(x_epoches, zeta_history, 'r-', label='Zeta')
    plt.title("Zeta Identification History")
    plt.xlabel("Epochs")
    plt.ylabel("Zeta Value")
    plt.grid(True, alpha=0.5)
    plt.legend()'''


if __name__ == '__main__':
    data=DataLoader()('ExperimentData.xlsx')
    epoches=5000
    model_trained, loss_trained,loss_history,zeta_history = train_pinn(data,epochs=epoches)
    plot_pinn_results(data, model_trained, loss_trained)