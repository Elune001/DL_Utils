from torch import nn

class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()
    def forward(self,out, targets,**kwargs):
        loss = 0 # compute loss and return
        return loss