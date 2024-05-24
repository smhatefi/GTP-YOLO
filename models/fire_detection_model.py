import torch
import torch.nn as nn
from models.yolo import Model

class LinearAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x

class GatedTemporaryPooling(nn.Module):
    def __init__(self, in_channels):
        super(GatedTemporaryPooling, self).__init__()
        self.gate = nn.Sigmoid()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        pooled = self.pool(x)
        gated = self.gate(pooled)
        return gated * pooled

class FireDetectionModel(Model):
    def __init__(self, cfg='models/yolov5s.yaml', ch=3, nc=None):
        super(FireDetectionModel, self).__init__(cfg, ch, nc)
        self.linear_attention = LinearAttention(in_channels=512, out_channels=512)
        self.gated_temporary_pooling = GatedTemporaryPooling(in_channels=512)

    def forward(self, x, augment=False, profile=False):
        x = super().forward(x, augment, profile)
        x = self.linear_attention(x)
        x = self.gated_temporary_pooling(x)
        return x

def load_pretrained_model():
    model = FireDetectionModel()
    model.load_state_dict(torch.load('yolov5s.pt', map_location='cpu')['model'])
    return model
