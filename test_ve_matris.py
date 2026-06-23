import torch
import torch.nn as nn
from torchvision import models

class MultimodalModel(nn.Module):
    def __init__(self, num_classes=5):
        super(MultimodalModel, self).__init__()
        self.cnn = models.resnet18(pretrained=False)
        self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 128)
        
        self.mlp = nn.Sequential(
            nn.Linear(3, 16), 
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU()
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, image, clinical_data):
        x1 = self.cnn(image)
        x2 = self.mlp(clinical_data)
        x = torch.cat((x1, x2), dim=1)
        return self.fusion(x)

print("🔬 GitHub Sürümü: Güncel model mimarisi başarıyla yüklendi.")
