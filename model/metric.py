import torch
import torch.nn.functional as F
import pysnooper

# @pysnooper.snoop(watch=("pred.shape", "pred", "target", "correct"))
def accuracy(output, target):
    with torch.no_grad():
        output = F.softmax(output, 1)
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        output = F.softmax(output, 1)
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def triplet_acc(hm, ml, hl):
    with torch.no_grad():
        bs = hm.shape[0]
        return torch.sum((hm < hl).long()) * 1.0 / bs
