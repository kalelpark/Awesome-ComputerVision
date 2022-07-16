import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

device = "cuda:0" if torch.cuda.is_available() else "cpu"

vgg_16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

class VGG16(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(VGG16, self).__init__()
        self.in_channels = in_channels

        self.feature_extractor = self.create_conv_layers(vgg_16)

        self.classification = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, out_features = n_classes)
        )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, 512*7*7)
        x = self.classification(x)

        return x
    
    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x
                
                layers += [nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                                     kernel_size= (3, 3), stride = (1, 1), padding = (1, 1)),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
                in_channels = x
            
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size= (2, 2), stride = (2, 2))]
        
        return nn.Sequential(**layers)