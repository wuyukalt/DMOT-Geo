import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .backbone import get_backbone
from .attention import get_attention
from .aggregation import get_aggregation


class OTGeo(nn.Module):
    def __init__(self, config):
        super(OTGeo, self).__init__()
        self.config = config
        self.backbone = get_backbone(backbone=config.backbone)
        self.attention = get_attention(attention=config.attention)
        self.aggregation = get_aggregation(aggregation=config.aggregation, num_channels=config.num_channels,
                                           num_clusters=config.num_clusters, cluster_dim=config.cluster_dim)

        # self.adaptivepool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # x(32,3,384,384)
        if self.config.attention.lower() == 'triplet':
            x = self.backbone(x)  # (32,384,24,24) (32,768,12,12)
            x = self.attention(x)  # (32,384,24,24)
            x = self.aggregation(x)  # (32,8192)

            # x = self.adaptivepool(x)
            # x = x.view(x.size(0), -1)  # (32,384)

        # return F.normalize(x.sum(dim=-1), p=2, dim=1).flatten(1)
        return F.normalize(x.flatten(1), p=2, dim=1)


class GeoModel(nn.Module):
    def __init__(self, config):
        super(GeoModel, self).__init__()
        self.model = OTGeo(config=config)
        if 'infonce' in config.loss.lower():
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, img1, img2=None):

        if img2 is not None:
            image_features1 = self.model(img1)
            image_features2 = self.model(img2)
            return image_features1, image_features2

        else:
            image_features = self.model(img1)
            return image_features
