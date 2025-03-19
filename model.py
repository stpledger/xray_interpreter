import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B1_Weights, EfficientNet_B6_Weights

class EfficientNetB1(nn.Module):
    def __init__(self, num_classes: int = 1000, fresh: bool = False):
        super(EfficientNetB1, self).__init__()
        if fresh:
            self.model = models.efficientnet_b1(weights=None)
        else:
            self.model = models.efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)

        if num_classes != 1000:
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x), {}
    

class EfficientNetB6(nn.Module):
    def __init__(self, num_classes: int = 1000, fresh: bool = False):
        super(EfficientNetB6, self).__init__()
        if fresh:
            self.model = models.efficientnet_b1(weights=None)
        else:
            self.model = models.efficientnet_b1(weights=EfficientNet_B6_Weights.DEFAULT)
            
        if num_classes != 1000:
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x), {}