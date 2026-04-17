import torch
import torchvision
from torch import nn
from thop import profile
import timm


class ConvNextTiny_0(nn.Module):
    def __init__(self):
        super().__init__()
        model = torchvision.models.convnext_tiny(weights='IMAGENET1K_V1')
        layers = list(model.features.children())
        self.backbone = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        return x


class ConvNextTiny_2(nn.Module):
    def __init__(self):
        super().__init__()
        model = torchvision.models.convnext_tiny(weights='IMAGENET1K_V1')
        layers = list(model.features.children())[:-2]
        self.backbone = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        return x


class ConvNextTiny_4(nn.Module):
    def __init__(self):
        super().__init__()
        model = torchvision.models.convnext_tiny(weights='IMAGENET1K_V1')
        layers = list(model.features.children())[:-4]
        self.backbone = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        return x


class ConvNextSmall_0(nn.Module):
    def __init__(self):
        super().__init__()
        model = torchvision.models.convnext_small(weights='IMAGENET1K_V1')
        layers = list(model.features.children())
        self.backbone = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        return x


class ConvNextSmall_2(nn.Module):
    def __init__(self):
        super().__init__()
        model = torchvision.models.convnext_small(weights='IMAGENET1K_V1')
        layers = list(model.features.children())[:-2]
        self.backbone = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        return x


class ConvNextSmall_4(nn.Module):
    def __init__(self):
        super().__init__()
        model = torchvision.models.convnext_small(weights='IMAGENET1K_V1')
        layers = list(model.features.children())[:-4]
        self.backbone = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        return x


class ConvNextBase_0(nn.Module):
    def __init__(self):
        super().__init__()
        model = torchvision.models.convnext_base(weights='IMAGENET1K_V1')
        layers = list(model.features.children())
        self.backbone = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        return x


class ConvNextBase_2(nn.Module):
    def __init__(self):
        super().__init__()
        model = torchvision.models.convnext_base(weights='IMAGENET1K_V1')
        layers = list(model.features.children())[:-2]
        self.backbone = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        return x


class ConvNextBase_4(nn.Module):
    def __init__(self):
        super().__init__()
        model = torchvision.models.convnext_base(weights='IMAGENET1K_V1')
        layers = list(model.features.children())[:-4]
        self.backbone = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        return x


class ResNet18_0(nn.Module):
    def __init__(self):
        super(ResNet18_0, self).__init__()
        model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        layers = list(model.children())[:-2]
        self.backbone = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        return x


class ResNet18_1(nn.Module):
    def __init__(self):
        super(ResNet18_1, self).__init__()
        model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        layers = list(model.children())[:-3]
        self.backbone = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        return x


class ResNet18_2(nn.Module):
    def __init__(self):
        super(ResNet18_2, self).__init__()
        model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        layers = list(model.children())[:-4]
        self.backbone = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        return x


class ResNet18_4(nn.Module):
    def __init__(self):
        super(ResNet18_4, self).__init__()
        model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        layers = list(model.children())[:-6]
        self.backbone = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        return x


class ResNet50_0(nn.Module):
    def __init__(self):
        super(ResNet50_0, self).__init__()
        model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
        layers = list(model.children())[:-2]
        self.backbone = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        return x


class ResNet50_2(nn.Module):
    def __init__(self):
        super(ResNet50_2, self).__init__()
        model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
        layers = list(model.children())[:-2]
        self.backbone = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        return x


class ResNet50_4(nn.Module):
    def __init__(self):
        super(ResNet50_4, self).__init__()
        model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
        layers = list(model.children())[:-4]
        self.backbone = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        return x


class ResNet101_0(nn.Module):
    def __init__(self):
        super(ResNet101_0, self).__init__()
        model = torchvision.models.resnet101(weights='IMAGENET1K_V1')
        layers = list(model.children())[:-2]
        self.backbone = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        return x


class ResNet101_1(nn.Module):
    def __init__(self):
        super(ResNet101_1, self).__init__()
        model = torchvision.models.resnet101(weights='IMAGENET1K_V1')
        layers = list(model.children())[:-3]
        self.backbone = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        return x


class ResNet101_2(nn.Module):
    def __init__(self):
        super(ResNet101_2, self).__init__()
        model = torchvision.models.resnet101(weights='IMAGENET1K_V1')
        layers = list(model.children())[:-4]
        self.backbone = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        return x


class ViTBase_0(nn.Module):
    def __init__(self):
        super(ViTBase_0, self).__init__()
        model = torchvision.models.vit_b_16(weights='IMAGENET1K_V1')
        self.conv_proj = model.conv_proj
        self.encoder = model.encoder
        self.class_token = model.class_token

    def forward(self, x):
        n = x.shape[0]
        x = self.conv_proj(x)
        h_out, w_out = x.shape[2], x.shape[3]
        x = x.reshape(n, 768, -1).permute(0, 2, 1)
        cls_token = self.class_token.expand(n, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.encoder(x)
        x = x[:, 1:, :]
        x = x.permute(0, 2, 1)
        x = x.reshape(n, 768, h_out, w_out)
        return x


class ViTSmall_0(nn.Module):
    def __init__(self):
        super(ViTSmall_0, self).__init__()
        self.model = timm.create_model('vit_small_patch16_384', pretrained=True)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = x[:, 1:, :].permute(0, 2, 1).reshape(-1, 384, 24, 24)

        return x


class ViTLarge_0(nn.Module):
    def __init__(self):
        super(ViTLarge_0, self).__init__()
        self.model = timm.create_model('vit_large_patch32_384', pretrained=True)
        self.num_patches = self.model.patch_embed.num_patches

    def forward(self, x):
        x = self.model.forward_features(x)
        B, L, C = x.shape
        patch_tokens = x[:, 1:, :]
        grid_size = int(self.num_patches ** 0.5)
        assert grid_size * grid_size == self.num_patches, "Number of patches is not a perfect square."
        x = patch_tokens.permute(0, 2, 1).view(-1, C, grid_size, grid_size)

        return x


class SwinBase_0(nn.Module):
    def __init__(self):
        super(SwinBase_0, self).__init__()
        self.model = timm.create_model('swin_base_patch4_window12_384', pretrained=True)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = x.permute(0, 3, 1, 2)
        return x


class Swinv2Base_0(nn.Module):
    def __init__(self):
        super(Swinv2Base_0, self).__init__()
        self.model = timm.create_model('swinv2_base_window12to24_192to384', pretrained=True)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = x.permute(0, 3, 1, 2)
        return x


class DINOv2Base_0(nn.Module):
    def __init__(self):
        super(DINOv2Base_0, self).__init__()
        self.model = timm.create_model('vit_base_patch14_dinov2', pretrained=True, img_size=384)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = x[:, 1:, :]  # [B, 729, 768]
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)

        return x

class DINOv2Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

    def forward(self, x):
        out_dict = self.model.forward_features(x)
        x = out_dict['x_norm_patchtokens']
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        # 转换为 (B, 768, H, W) 格式以适配后续卷积层
        x = x.permute(0, 2, 1).reshape(B, C, H, W)

        return x


def get_backbone(backbone='ConvNeXt'):
    if backbone.lower() == 'convnexttiny_0':
        return ConvNextTiny_0()
    elif backbone.lower() == 'convnexttiny_2':
        return ConvNextTiny_2()
    elif backbone.lower() == 'convnexttiny_4':
        return ConvNextTiny_4()
    elif backbone.lower() == 'convnextsmall_0':
        return ConvNextSmall_0()
    elif backbone.lower() == 'convnextsmall_2':
        return ConvNextSmall_2()
    elif backbone.lower() == 'convnextsmall_4':
        return ConvNextSmall_4()
    elif backbone.lower() == 'convnextbase_4':
        return ConvNextBase_4()
    elif backbone.lower() == 'convnextbase_2':
        return ConvNextBase_2()
    elif backbone.lower() == 'convnextbase_0':
        return ConvNextBase_0()
    elif backbone.lower() == 'swinv2base_0':
        return Swinv2Base_0()
    elif backbone.lower() == 'swinbase_0':
        return SwinBase_0()
    elif backbone.lower() == 'resnet101_0':
        return ResNet101_0()
    elif backbone.lower() == 'resnet50_4':
        return ResNet50_4()
    elif backbone.lower() == 'resnet50_2':
        return ResNet50_2()
    elif backbone.lower() == 'resnet50_0':
        return ResNet50_0()
    elif backbone.lower() == 'resnet18_4':
        return ResNet18_4()
    elif backbone.lower() == 'resnet18_2':
        return ResNet18_2()
    elif backbone.lower() == 'resnet18_1':
        return ResNet18_1()
    elif backbone.lower() == 'resnet18_0':
        return ResNet18_0()
    elif backbone.lower() == 'vitlarge_0':
        return ViTLarge_0()
    elif backbone.lower() == 'vitbase_0':
        return ViTBase_0()
    elif backbone.lower() == 'vitsmall_0':
        return ViTSmall_0()
    elif backbone.lower() == 'dinov2base_0':
        return DINOv2Base_0()
    elif backbone.lower() == 'dinov2base':
        return DINOv2Base()

if __name__ == '__main__':
    convnext = ConvNextSmall_2()
    x = torch.randn(1, 3, 384, 384)
    flops, params = profile(convnext, inputs=(x,))
    print(f"FLOPs: {flops / 1e9:.2f}G, Params: {params / 1e6:.2f}M")
