import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34


class ProjectionHead(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=2048, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ResNet34Encoder(nn.Module):
    def __init__(self, projection_dim=128, projection_hidden_dim=2048):
        super().__init__()
        backbone = resnet34(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = backbone.fc.in_features
        self.projector = ProjectionHead(self.feature_dim, projection_hidden_dim, projection_dim)

    def forward_features(self, x):
        h = self.features(x).flatten(1)
        return h

    def forward_projected(self, x):
        h = self.forward_features(x)
        z = self.projector(h)
        return F.normalize(z, dim=1)

    def forward(self, x, projected=True):
        if projected:
            return self.forward_projected(x)
        return self.forward_features(x)


class LinearProbe(nn.Module):
    def __init__(self, encoder, feature_dim=512, num_classes=10):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(feature_dim, num_classes)
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            h = self.encoder.forward_features(x)
        return self.classifier(h)
