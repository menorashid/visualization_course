from torchvision import models
import torch.nn as nn


class ReLUM(nn.Module):

    def __init__(self,n_classes):
        super(Cifar_10_Alex, self).__init__()
        self.num_classes = n_classes
        self.features = []
        
        self.features.append(nn.Conv2d(3, 32, 5, stride = 1, padding = 2))
        self.features.append(nn.ReLU(True))
        self.features.append(nn.MaxPool2d(3,2))

        self.features.append(nn.Conv2d(32, 32, 5, stride = 1, padding = 2))
        self.features.append(nn.ReLU(True))
        self.features.append(nn.MaxPool2d(3,2))
        
        self.features.append(nn.Conv2d(32, 64, 5, stride = 1, padding = 2))
        self.features.append(nn.ReLU(True))
        self.features.append(nn.MaxPool2d(3,2))
        
        self.features = nn.Sequential(*self.features)
        
        self.classifier = []
        self.classifier.append(nn.Linear(64*3*3,self.num_classes))
        self.classifier = nn.Sequential(*self.classifier)

        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 64*3*3)
        x = self.classifier(x)
        return x