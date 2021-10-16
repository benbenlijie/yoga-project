import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import math
from torchvision.models.mobilenetv3 import mobilenet_v3_large

class MobileNetModel(BaseModel):
    def __init__(self, feature_dim, num_classes, s=30.0, m=0.5, 
        easy_margin=False):
        super().__init__()
        backbone = mobilenet_v3_large(True)
        dim_feature_backbone = 960
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.features1 = nn.Sequential(
            backbone.features,
            backbone.avgpool,
            nn.Flatten(1),
        )
        self.features2 = nn.Sequential(
            nn.Linear(dim_feature_backbone, self.feature_dim),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        self.arc_margin = ArcMarginProduct(self.feature_dim, self.num_classes, s, m, easy_margin)

        for m in self.features2.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, target=None):
        feature = self.features1(x)
        feature = self.features2(feature)
        
        if target != None:
            output = self.arc_margin(feature, target)
        return output

class ArcMarginProduct(BaseModel):
    def __init__(self, in_features, out_features, s=30.0, m=0.5, 
        easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.easy_margin = easy_margin
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self.weight = torch.nn.Parameter(torch.FloatTensor(self.out_features, self.in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, feature, label):
        cosine = F.linear(F.normalize(feature), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=feature.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output
        
