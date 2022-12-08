import torch
from torch import nn
class My_model(nn.Module):
    def __init__(self):
        super(My_model, self).__init__()
    def forward(self,x):
        return