import torch
import torch.nn as nn

class Depthwise(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()
        
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride = stride, 
                      padding = 1, groups = in_channels, bias = False),
            # Groups로 channels의 Group을 지정하는 것이 가능합니다.
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x


class MobileNet(nn.Module):
    # MobileNet에는 하이퍼파라미터가 존재합니다. (채널 수를 조절하는 
    # Width_Multiplier 파라미터를 신경써야합니다.)
    def __init__(self, width_multiplier, num_classes = 10, init_weights = True):
        super().__init__()
        self.init_weights = init_weights
        alpha = width_multiplier

        self.conv1 = BasicConv2d(3, int(32*alpha), 3, stride = 2, padding = 1)
        self.conv2 = Depthwise(int(32*alpha), int(64*alpha), stride = 1)

        # down sample
        self.conv3 = nn.Sequential(
            Depthwise(int(64*alpha), int(128*alpha), stride = 2),
            Depthwise(int(128*alpha), int(128*alpha), stride = 1)
        )

        self.conv4 = nn.Sequential(
            Depthwise(int(128*alpha), int(256*alpha), stride = 2),
            Depthwise(int(256*alpha), int(256*alpha), stride = 1)
        )


        self.conv5 = nn.Sequential(
            Depthwise(int(256*alpha), int(512*alpha), stride=2),
            Depthwise(int(512*alpha), int(512*alpha), stride=1),
            Depthwise(int(512*alpha), int(512*alpha), stride=1),
            Depthwise(int(512*alpha), int(512*alpha), stride=1),
            Depthwise(int(512*alpha), int(512*alpha), stride=1),
            Depthwise(int(512*alpha), int(512*alpha), stride=1),
        )

        self.conv6 = nn.Sequential(
            Depthwise(int(512*alpha), int(1024*alpha), stride = 2)
        )

        self.conv7 = nn.Sequential(
            Depthwise(int(1024*alpha), int(1024*alpha), stride = 2)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(int(1024*alpha), num_classes)

        if self.init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x  

    
    # weights initialization function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def mobilenet(alpha=1, num_classes=10):
    return MobileNet(alpha, num_classes)

x = torch.randn((3, 3, 224, 224))
model = mobilenet(alpha=1)
output = model(x)
print('output size:', output.size())