from base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class GraphConvolution(nn.Module):
    def __init__(self, in_c, out_c, bias=False):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.weight = nn.Parameter(torch.Tensor(in_c, out_c))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_c))
        else:
            self.register_parameter("bias", None)
        self.init_parameters()
    
    def init_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output
        

class GCNModel(BaseModel):
    def __init__(self, )