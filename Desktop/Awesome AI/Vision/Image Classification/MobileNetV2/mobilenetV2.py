import torch.nn as nn
import math
import numpy as np

def conv_bn(input_channel, output_channel, stride):
    return nn.Sequential(
        nn.Conv2d(input_channel, output_channel, 3, stride, 1, bias = False),
        nn.BatchNorm2d(output_channel),
        nn.ReLU6(inplace = True)
    )

def conv_1x1_bn(input_channel, output_channel):
    return nn.Sequential(
        nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias = False),
        nn.BatchNorm2d(output_channel),
        nn.ReLU6(inplace = True)
    )

def make_divisible(x, divisible_by = 8):
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, input_channel, output_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(input_channel * expand_ratio)
        self.use_res_connect = self.stride == 1 and input_channel == output_channel

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups = hidden_dim, bias = False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace = True),
                nn.Conv2d(hidden_dim, output_channel, 1, 1, 0, bias = False),
                nn.BatchNorm2d(output_channel),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(input_channel, hidden_dim, 1, 1, 0, bias = False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace = True),

                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups = hidden_dim, bias = False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace = True),
                nn.Conv2d(hidden_dim, output_channel, 1, 1, 0, bias = False),
                nn.BatchNorm2d(output_channel),
            )
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, n_classes = 1000, input_size =224, width_mult = 1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # expansion_factor = t, channel = c, range = n, stride = s
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        assert input_size % 32 == 0
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]

        for t, c, n, s in inverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio = t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio = t))
                input_channel = output_channel
        
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Linear(self.last_channel, n_classes)

        self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x 

    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def mobilenet_v2(pretrained=True):
    model = MobileNetV2(width_mult=1)
    return model

# !pip install torchsummary
import torch
from torchsummary import summary

model = mobilenet_v2()
input = torch.randn(3, 224, 224).unsqueeze(0)
output = model(input)
output.size()

summary(model, (3, 224, 224))

