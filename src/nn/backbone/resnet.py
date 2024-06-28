import torch
import torch.nn as nn 
import torch.nn.functional as F 
from core import register

# __all__ = ['resnet18','resnet34','resnet50','resnet101','resnet152']
__all__ = ['Resnet50']
from torchvision.models import resnet50



#   depth: Resnet50
#   return_idx: [1, 2, 3]
#   num_stages: 4
#   freeze_norm: True
#   pretrained: True

@register
class Resnet50(nn.Module):
    __inject__ = ['ResNet50', ]
    def __init__(self,**kwargs):
        super().__init__()
        if kwargs['pretrained'] ==True:
            weights="IMAGENET1K_V2"
        else:
            weights=None
        self.pretrain = resnet50(weights)
    def forward(self,x):
        x = self.pretrain(x)
        return x
