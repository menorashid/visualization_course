from torchvision import models
import torch.nn as nn
from ReLUM_simple import *
class Cifar_10_Alex(nn.Module):

    def __init__(self,n_classes):
        super(Cifar_10_Alex, self).__init__()
        self.num_classes = n_classes
        self.features = []
        
        self.features.append(nn.Conv2d(3, 32, 5, stride = 1, padding = 2))

        # self.features.append(nn.ReLU(True))
        self.features.append(nn.MaxPool2d(3,2))
        self.features.append(ReLUM(32,32,5,1,2))

        # self.features.append(nn.ReLU(True))
        self.features.append(nn.MaxPool2d(3,2))
        self.features.append(ReLUM(32,32,5,1,2))

        # self.features.append(nn.ReLU(True))
        self.features.append(nn.MaxPool2d(3,2))
        self.features.append(ReLUM(32,64,5,1,2))

        self.features = nn.Sequential(*self.features)
        
        self.classifier = []
        self.classifier.append(nn.Linear(64*3*3,self.num_classes))
        self.classifier = nn.Sequential(*self.classifier)

        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 64*3*3)
        x = self.classifier(x)
        return x

class Network:
    def __init__(self,n_classes = 10, init = True):
        model = Cifar_10_Alex(n_classes)
        if init:
            for idx_m,m in enumerate(model.modules()): 
                if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear):    
                    nn.init.xavier_normal(m.weight.data)
                    nn.init.constant(m.bias.data,0.)

        self.model = model



    def get_lr_list(self, lr):
        lr_list= [{'params': self.model.features.parameters(), 'lr': lr[0]}]\
                +[{'params': self.model.classifier.parameters(), 'lr': lr[1]}]
        return lr_list


def main():
    import numpy as np
    import torch
    from torch.autograd import Variable

    net = Network(8)
    print net.model
    input = np.zeros((10,3,32,32))
    input = torch.Tensor(input)
    print input.shape
    input = Variable(input)
    output = net.model(input)
    print output.data.shape


if __name__=='__main__':
    main()
