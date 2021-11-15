import torch.nn.functional as F
import torch
import pysnooper

def nll_loss(output, target):
    return F.nll_loss(output, target)

# @pysnooper.snoop(watch="target")
def crossentropy_loss(output, target):
    return F.cross_entropy(output, target)

# @pysnooper.snoop()
def arc_margin_loss(output, target):
    output_shape = output.shape
    loss = F.softmax(output, dim=1)
    loss_shape = loss.shape
    loss = loss[torch.arange(loss.size(0)), target]
    loss_shape = loss.shape
    loss = torch.mean(loss)
    return loss


def triple_loss(hm, ml, hl, margin):
    return torch.mean(torch.clamp(hm-hl+margin, min=0)+torch.clamp(ml-hl+margin, min=0))

def triple_loss_half(hm, ml, hl, margin):
    return torch.mean(torch.clamp(hm-hl+margin, min=0))

