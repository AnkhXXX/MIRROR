from torch import nn
import torch
import torch.nn.functional as F


class MFACConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        ]
        super(MFACConv, self).__init__(*modules)


class MFACPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(MFACPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(MFACPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class MFAC(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, ):
        super(MFAC, self).__init__()
        out_channels = out_channels
        modules = []
        modules.append(nn.Sequential(

            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()))

        for rate in atrous_rates:
            modules.append(MFACConv(in_channels, out_channels, rate))

        modules.append(MFACPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
