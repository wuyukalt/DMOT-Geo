import torch
import torchvision
from torch import nn
from thop import profile


class ConvNextTiny(nn.Module):
    """FLOPs: 10.89G, Params: 12.34M"""

    def __init__(self):
        super().__init__()
        model = torchvision.models.convnext_tiny(weights='IMAGENET1K_V1')
        # layers = list(model.features.children())[:-4]
        layers = list(model.features.children())[:-2]
        # layers = list(model.features.children())
        self.backbone = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.backbone(x)  # (B, 768, H/16, W/16)


class ConvNextSmall(nn.Module):
    def __init__(self):
        super().__init__()
        model = torchvision.models.convnext_small(weights='IMAGENET1K_V1')
        # layers = list(model.features.children())[:-4]
        # layers = list(model.features.children())[:-2]
        layers = list(model.features.children())
        self.backbone = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.backbone(x)  # (B, 768, H/16, W/16)

class ConvNextBase(nn.Module):
    def __init__(self):
        super().__init__()
        model = torchvision.models.convnext_base(weights='IMAGENET1K_V1')
        layers = list(model.features.children())[:-2]
        self.backbone = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.backbone(x)  # (B, 768, H/8, W/8)


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        layers = list(model.children())[:-2]
        self.backbone = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.backbone(x)  # (B, 512, H/32, W/32)


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
        layers = list(model.children())[:-2]
        self.backbone = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        return x  # (B, 2048, H/32, W/32)


def get_backbone(backbone='ConvNeXt'):
    if backbone.lower() == 'convnexttiny':
        return ConvNextTiny()
    elif backbone.lower() == 'convnextsmall':
        return ConvNextSmall()
    elif backbone.lower() == 'convnextbase':
        return ConvNextBase()
    elif backbone.lower() == 'resnet50':
        return ResNet50()
    else:
        return ResNet18()


if __name__ == '__main__':
    convnext = ConvNextSmall()
    x = torch.randn(1, 3, 384, 384)
    flops, params = profile(convnext, inputs=(x,))
    print(f"FLOPs: {flops / 1e9:.2f}G, Params: {params / 1e6:.2f}M")
