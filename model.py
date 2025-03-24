import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    EfficientNet_B2_Weights,
    EfficientNet_B3_Weights,
    EfficientNet_B4_Weights,
    EfficientNet_B5_Weights,
    EfficientNet_B6_Weights,
    EfficientNet_B7_Weights
)

class EfficientNet(nn.Module):
    def __init__(self, variant: str, num_classes: int = 1000, fresh: bool = False):
        super().__init__()
        variant = variant.lower()
        mapping = {
            "b0": (models.efficientnet_b0, EfficientNet_B0_Weights),
            "b1": (models.efficientnet_b1, EfficientNet_B1_Weights),
            "b2": (models.efficientnet_b2, EfficientNet_B2_Weights),
            "b3": (models.efficientnet_b3, EfficientNet_B3_Weights),
            "b4": (models.efficientnet_b4, EfficientNet_B4_Weights),
            "b5": (models.efficientnet_b5, EfficientNet_B5_Weights),
            "b6": (models.efficientnet_b6, EfficientNet_B6_Weights),
            "b7": (models.efficientnet_b7, EfficientNet_B7_Weights)
        }
        if variant not in mapping:
            raise ValueError(f"Unsupported EfficientNet variant: {variant}")
        
        builder, weights_cls = mapping[variant]
        if fresh:
            self.model = builder(weights=None)
        else:
            self.model = builder(weights=weights_cls.DEFAULT)
        
        if num_classes != 1000:
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x), {}

# Optional alias classes for backward compatibility
class EfficientNetB0(EfficientNet):
    def __init__(self, num_classes: int = 1000, fresh: bool = False):
        super().__init__("b0", num_classes, fresh)

class EfficientNetB1(EfficientNet):
    def __init__(self, num_classes: int = 1000, fresh: bool = False):
        super().__init__("b1", num_classes, fresh)

class EfficientNetB2(EfficientNet):
    def __init__(self, num_classes: int = 1000, fresh: bool = False):
        super().__init__("b2", num_classes, fresh)

class EfficientNetB3(EfficientNet):
    def __init__(self, num_classes: int = 1000, fresh: bool = False):
        super().__init__("b3", num_classes, fresh)

class EfficientNetB4(EfficientNet):
    def __init__(self, num_classes: int = 1000, fresh: bool = False):
        super().__init__("b4", num_classes, fresh)

class EfficientNetB5(EfficientNet):
    def __init__(self, num_classes: int = 1000, fresh: bool = False):
        super().__init__("b5", num_classes, fresh)

class EfficientNetB6(EfficientNet):
    def __init__(self, num_classes: int = 1000, fresh: bool = False):
        super().__init__("b6", num_classes, fresh)

class EfficientNetB7(EfficientNet):
    def __init__(self, num_classes: int = 1000, fresh: bool = False):
        super().__init__("b7", num_classes, fresh)