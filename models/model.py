import torch.nn as nn
from .backbone import get_backbone
from .attention import get_attention
from .aggregation import get_aggregation
from utils.losses import CrossViewOTLoss


class DMOTGeo(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.backbone = get_backbone(backbone=config.backbone)
        self.attention = get_attention(attention=config.attention)
        self.aggregation = get_aggregation(aggregation=config.aggregation,
                                           in_channels=config.num_channels,
                                           cluster_dim=config.cluster_dim,
                                           num_clusters=config.num_clusters,
                                           num_scales=config.num_scales,
                                           sinkhorn_iters=config.sinkhorn_iters)

        self.cv_ot_loss = CrossViewOTLoss(
            in_channels=config.cluster_dim,
            hidden_dim=config.num_clusters,
            sinkhorn_iters=config.sinkhorn_iters,
            temperature=config.temperature
        )

    def extract_featmap(self, x):
        feat = self.backbone(x)
        feat = self.attention(feat)
        return feat

    def encode_image(self, x):
        """
        测试阶段调用这个函数
        输入单张图，输出全局描述子
        """
        feat = self.extract_featmap(x)
        feat = self.aggregation(feat, return_local=False)
        return feat

    def forward_train(self, drone, satellite):
        """
        训练阶段调用
        """
        feat_d = self.extract_featmap(drone)
        feat_s = self.extract_featmap(satellite)

        drone_desc, local_d = self.aggregation(feat_d, return_local=True)
        sat_desc, local_s = self.aggregation(feat_s, return_local=True)

        loss_ot = self.cv_ot_loss(local_d, local_s)

        return drone_desc, sat_desc, loss_ot
