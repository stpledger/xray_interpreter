import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B1_Weights

class EfficientNetB1(nn.Module):
    def __init__(self, num_classes=1000):
        super(EfficientNetB1, self).__init__()
        self.model = models.efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
        if num_classes != 1000:
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x), {}