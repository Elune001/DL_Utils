import numpy as np
import torch.nn.functional as F


class Metric():
    def __init__(self):
        self.MAE_list = []
        self.MSE_list = []
        self.RMSE_list = []

    def __call__(self, outputs,targets):
        mse = F.mse_loss(outputs,targets,reduction='mean')
        rmse = mse.sqrt()
        mae = F.l1_loss(outputs,targets,reduction='mean')
        self.MSE_list.append(mse.cpu().numpy())
        self.MAE_list.append(mae.cpu().numpy())
        self.RMSE_list.append(rmse.cpu().numpy())

        return np.asarray(self.MSE_list).mean(), np.asarray(self.RMSE_list).mean(), np.asarray(self.MAE_list).mean()