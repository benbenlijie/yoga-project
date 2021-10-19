import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel=(1,1), 
                stride=(1,1), padding=(0,0), groups=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel, groups=groups,
                stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(num_features=out_channel),
            nn.PReLU(out_channel)
        )
    
    def forward(self, x):
        return self.layers(x)

class DepthWise(nn.Module):
    def __init__(self, in_channel, out_channel, residual=False, kernel=(3,3),
                stride=(2,2), padding=(1,1), groups=1):
        super().__init__()
        

class Yoga2dConvModel(BaseModel):
    def __init__(self, num_classes=100):
        pass
    
    def forward(self, x):
        pass
    