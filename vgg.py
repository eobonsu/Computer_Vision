'''
Modified from https://github.com/pytorch/vision.git
'''
import math
import torch
import torch.nn as nn
import torch.nn.init as init

# ---- new imports at top ----
from collections import OrderedDict
try:
    from antialiased_cnns import BlurPool   # pip install antialiased-cnns
except ImportError:
    # lightweight, no‑dep fallback
    class BlurPool(nn.AvgPool2d):
        def __init__(self, channels, filt_size=3, stride=2):
            super().__init__(kernel_size=stride, stride=stride)

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):
    ''' VGG model '''
    def __init__(self, features, out_channels):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, out_channels),
        )

        ''' Initialize weights '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(cfg, in_dims=3, batch_norm=False, aa_downsample=True):
    """
    cfg list        : as before
    aa_downsample   : when True, replaces every MaxPool with BlurPool
    """

    layers = []
    in_channels = in_dims

    for v in cfg:
        if v == 'M':                     # down‑sampling block
            if aa_downsample:
                layers += [BlurPool(in_channels, stride=2)]
            else:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=not batch_norm)
            norm = nn.GroupNorm(8, v) if batch_norm else nn.Identity()
            layers += [conv2d, norm, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


    """for v in cfg:
        if v == 'M':        # down‑sampling block
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)"""


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# def vgg11(out_channels=10):
#     """VGG 11-layer model (configuration "A")"""
#     return VGG(make_layers(cfg['A']), out_channels)
# 
# 
def vgg11_bn(in_dims=3, out_dims=10):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], in_dims, batch_norm=True), out_dims)
# 
# 
# def vgg13(out_channels=10):
#     """VGG 13-layer model (configuration "B")"""
#     return VGG(make_layers(cfg['B']), out_channels)
# 
# 
# def vgg13_bn(out_channels=10):
#     """VGG 13-layer model (configuration "B") with batch normalization"""
#     return VGG(make_layers(cfg['B'], batch_norm=True), out_channels)


# def vgg16(out_dims=10):
#     """VGG 16-layer model (configuration "D")"""
#     return VGG(make_layers(cfg['D']), out_dims)


def vgg16_bn(in_dims=3, out_dims=10):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], in_dims, batch_norm=True), out_dims)


# def vgg19(out_channels=10):
#     """VGG 19-layer model (configuration "E")"""
#     return VGG(make_layers(cfg['E']), out_channels)


def vgg19_bn(in_dims=3, out_dims=10):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], in_dims, batch_norm=True), out_dims)
