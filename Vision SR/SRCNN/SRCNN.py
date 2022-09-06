import math
import torch
import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self, num_channels = 3, feature_dim = 64, map_dim = 32):
        super(SRCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(num_channels, feature_dim, (9, 9), (1, 1), (4, 4)),
            nn.ReLU(True)
        )

        self.map = nn.Sequential(
            nn.Conv2d(feature_dim, map_dim, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True)
        )

        self.reconstruction = nn.Conv2d(map_dim, num_channels, (5, 5), (1, 1), (2, 2))
        self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        out = self.map(out)
        out = self.reconstruction(out)

        return out
            
    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, 0.0, math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                nn.init.zeros_(module.bias.data)
        nn.init.normal_(self.reconstruction.weight.data, 0.0, 0.001)
        nn.init.zeros_(self.reconstruction.bias.data)