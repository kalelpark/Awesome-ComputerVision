import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
  def __init__(self, n_classes):
    super(AlexNet, self).__init__()
    
    self.feature_extractor = nn.Sequential(
      # First Layer
      nn.Conv2d(in_channels = 3, out_channels = 96, kernel_size = 11, stride = 4),
      nn.ReLU(inplace = True),
      nn.LocalResponseNorm(size = 5, alpha = 1e-3, beta = 0.75, k = 2),
      nn.MaxPool2d(kernel_size = 3, stride = 2),

      # Second Layer
      nn.Conv2d(in_channels = 96, out_channels = 256, kernel_size = 5,stride = 1, padding = 2),
      nn.ReLU(),
      nn.LocalResponseNorm(size = 5, alpha = 1e-4, beta = 0.75, k = 2),
      nn.MaxPool2d(kernel_size = 3, stride = 2),

      # Third Layer
      nn.Conv2d(in_channels = 256, out_channels = 384, kernel_size = 3, stride = 1, padding = 1),
      nn.ReLU(),

      # Fourth Layer
      nn.Conv2d(in_channels = 384, out_channels = 384, kernel_size = 3, stride = 1, padding = 1),
      nn.ReLU(),

      nn.Conv2d(in_channels = 384, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
      nn.ReLU(),
      nn.MaxPool2d(3, 2),
    )

    self.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(256 * 6 * 6, out_features = 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(in_features = 4096, out_features = 4096),
        nn.ReLU(),
        nn.Linear(in_features = 4096, out_features = n_classes),
    )

  def init_weight(self):
    for layer in self.feature_extractor:
      if isinstance(layer, nn.Conv2d):
        nn.init.normal_(layer.weight, mean = 0, std = 0.01)
        nn.init.constant_(layer.bias, 0)
    nn.init.constant_(self.net[4].bias, 1)
    nn.init.constant_(self.net[10].bias, 1)
    nn.init.constant_(self.net[12].bias, 1)
  
  def forward(self , x):
    x = self.feature_extractor(x)
    x = x.view(-1, 256*6*6)
    x = self.classifier(x)

    return x