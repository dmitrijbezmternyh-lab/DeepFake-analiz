import torch.nn as nn
from torchvision import models
from .highpass_filter import HighPassFilter

class DeepFakeDetector(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.filter = HighPassFilter()
        self.backbone = models.efficientnet_b0(weights=None)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.filter(x)
        return self.backbone(x).squeeze()
