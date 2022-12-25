# !pip install torchsummary

import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import torch.optim as optim 
from torchsummary import summary
import math

class fire(nn.Module):
    def __init__(self, in_channels, out_channels, expand_channels):
        super(fire, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, expand_channels, kernel_size = 1, stride = 1)
        self.bn2 = nn.BatchNorm2d(expand_channels)
        self.conv3 = nn.Conv2d(out_channels, expand_channels, kernel_size = 3, stride = 1, padding = 1)
        self.bn3 = nn.BatchNorm2d(expand_channels)
        self.relu2 = nn.ReLU(inplace = True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2./n))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        out1 = self.conv2(x)
        out1 = self.bn2(out1)
        out2 = self.conv3(x)
        out2 = self.bn3(out2)
        out = torch.cat([out1, out2], 1)
        out = self.relu2(out)

        return out         

class SqueezeNet(nn.Module):
    def __init__(self):
        super(SqueezeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size = 3, stride = 1, padding = 1)
        self.bn1 = nn.BatchNorm2d(96)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.fire2 = fire(96, 16, 64)
        self.fire3 = fire(128, 16, 64)
        self.fire4 = fire(128, 32, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2 , stride = 2)
        self.fire5 = fire(256, 32, 128)
        self.fire6 = fire(256, 48, 192)
        self.fire7 = fire(384, 48, 192)
        self.fire8 = fire(384, 64, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.fire9 = fire(512, 64, 256)
        self.conv2 = nn.Conv2d(512, 1000, kernel_size = 1, stride =1 )
        self.avg_pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(1000, 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.maxpool2(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.maxpool3(x)
        x = self.fire9(x)
        x = self.conv2(x)
        x = self.avg_pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
        
def squeezenet(pretrained=False):
    net = SqueezeNet()

    return net


# model = squeezenet()
# input = torch.randn(3, 224, 224).unsqueeze(0)
# output = model(input)
# output.size()

# summary(model,(3, 224, 224))