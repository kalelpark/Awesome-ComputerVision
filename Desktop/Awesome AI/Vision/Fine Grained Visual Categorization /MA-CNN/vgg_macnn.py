import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):
    def __init__(self, features, num_classes, init_weights = True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        if init_weights:
            self.initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = "fan_out", nonlinearity = "relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
def make_layers(cfg, batch_norm = False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)]
        else:
            conv2d = nn.Conv2d(in_channels, v , kernel_size = 3, padding = 1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace = True)]
            else:
                layers += [conv2d, nn.ReLU(inplace = True)]
            in_channels = v
            
    return nn.Sequential(*layers)

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm = batch_norm), **kwargs)
    
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress = progress)
        model.load_state_dict(state_dict, strict = False)
    
    return model

def vgg11(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)


def vgg11_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)


def vgg13(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)


def vgg13_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13_bn', 'B', True, pretrained, progress, **kwargs)


def vgg16(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)


def vgg19(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)


def vgg19_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SELayer(nn.Module):
    def __init__(self, channel, reduction = 1):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias = False),
            nn.Tanh(),
            nn.Linear(channel // reduction, channel, bias = False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, h, w = x.size()
        print(x.size())
        y = self.avg_pool(x).view(b, c)
        print(y.size())
        y = self.fc(y)
        print(y.size())
        y_ = y.view(b, c, 1, 1)
        print(y_.size())
        M = torch.sum(x * y_.expand_as(x), dim = 1, keepdim = True)
        print(M.size())
        M = F.normalize(M.view(b, -1), dim = -1, p = 2).view(b, 1, h, w)
        print(M.size())
        P = x * M.expand_as(x)
        print(P.size())
        P = F.avg_pool2d(P, (h, w))
        print(P.size())
        return P, M.squeeze(dim = 1), y

class MACNN(nn.Module):
    def __init__(self):
        super(MACNN, self).__init__()
        self.vgg = vgg19(True)
        self.feat_dims = 512
        self.se1 = SELayer(self.feat_dims)
        self.se2 = SELayer(self.feat_dims)
        self.se3 = SELayer(self.feat_dims)
        self.se4 = SELayer(self.feat_dims)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.cnnfc = nn.Linear(self.feat_dims, 200)
        
        self.fc1 = nn.Linear(self.feat_dims, 200)
        self.fc2 = nn.Linear(self.feat_dims, 200)
        self.fc3 = nn.Linear(self.feat_dims, 200)
        self.fc4 = nn.Linear(self.feat_dims, 200)
        self.fcall = nn.Linear(5*self.feat_dims, 200)
    
    def forward(self, x):
        feat_maps = self.vgg(x)
        
        cnn_pred = self.cnnfc(self.pool(feat_maps).flatten(1))
        P1, M1, y1 = self.se1(feat_maps.detach())
        P2, M2, y2 = self.se2(feat_maps.detach())
        P3, M3, y3 = self.se3(feat_maps.detach())
        P4, M4, y4 = self.se4(feat_maps.detach())
        
        pred1 = self.fc1(P1.flatten(1))
        pred2 = self.fc2(P1.flatten(1))
        pred3 = self.fc3(P1.flatten(1))
        pred4 = self.fc4(P1.flatten(1))
        P = torch.cat([P1, P2, P3, P4, self.pool(feat_maps)], dim = 1)
        pred = self.fcall(P.flatten(1))
        
        return feat_maps,cnn_pred,\
               [P1,P2,P3,P4],\
               [M1,M2,M3,M4],\
               [y1,y2,y3,y4],\
               [pred1,pred2,pred3,pred4,pred]