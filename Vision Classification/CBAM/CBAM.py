import torch
import torch.nn as nn
from torchsummary import summary


class ResPourModel(nn.Module):
    def __init__(self):
        super(ResPourModel, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size = 1)
        self.maxpool = nn.AdaptiveMaxPool2d(output_size = 1)

        self.fc = nn.Sequential(
            nn.Conv2d(2048, 128, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(128, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        self.fs = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size = (7, 7), stride = (1, 1), padding = (3, 3), bias = False),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(2048, 80)
    
    def forward(self, x):
        feature_x, avg_x, max_x = self.feature_extractor(x), self.avgpool(x), self.maxpool(x)
        # channel_attention
        avgout, maxout = self.fc(self.avgpool(avg_x)), self.fc(self.maxpool(max_x))
        sig_out = self.sigmoid(avgout + maxout)

        # spatial_attention
        out = sig_out * feature_x
        avgout = torch.mean(out, dim = 1, keepdim = True) 
        maxout, _ = torch.max(out, dim = 1, keepdim = True)
        x = torch.cat([avgout, maxout], dim = 1)
        x = self.fs(x)
        x = self.sigmoid(x) * out
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

img = torch.randn(2048, 7, 7).unsqueeze(0)
model = ResPourModel()
output = model(img)
output.size()