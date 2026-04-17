import torch
import torch.nn as nn
import torch.distributed.nn
import torch.nn.functional as F
from models.aggregation import log_cross_view_ot


class InfoNCE(nn.Module):

    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()

        self.loss_function = loss_function
        self.device = device

    def forward(self, image_features1, image_features2, logit_scale):
        # (32,8192) (32,8192)
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)

        # (32,32) (32,32)
        logits_per_image1 = logit_scale * image_features1 @ image_features2.T
        logits_per_image2 = logits_per_image1.T

        # (0,1,2,3,....31)
        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
        loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels)) / 2
        # 3.4691
        return loss

class CrossViewOTLoss(nn.Module):
    def __init__(self, in_channels=128, hidden_dim=128, sinkhorn_iters=20, temperature=0.1):
        super().__init__()

        self.sinkhorn_iters = sinkhorn_iters
        self.temperature = temperature

        # 动态 dustbin，训练时辅助分支单独预测
        self.dustbin_head = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, feat_d, feat_s):

        dustbin = self.dustbin_head(feat_d)  # (B, 1, Nd)

        # 如果 Nd != Ns，插值到 Ns
        if dustbin.shape[-1] != feat_s.shape[-1]:
            dustbin = F.interpolate(dustbin, size=feat_s.shape[-1], mode='linear', align_corners=False)

        log_T_ds = log_cross_view_ot(
            feat_q=feat_d,
            feat_k=feat_s,
            dustbin=dustbin,
            temperature=self.temperature,
            iters=self.sinkhorn_iters
        )  # (B, Nd+1, Ns)

        log_T_sd = log_cross_view_ot(
            feat_q=feat_s,
            feat_k=feat_d,
            dustbin=F.interpolate(
                self.dustbin_head(feat_s),
                size=feat_d.shape[-1],
                mode='linear',
                align_corners=False
            ),
            temperature=self.temperature,
            iters=self.sinkhorn_iters
        )  # (B, Ns+1, Nd)

        T_ds = torch.exp(log_T_ds[:, :-1, :])  # (B, Nd, Ns)
        T_sd = torch.exp(log_T_sd[:, :-1, :])  # (B, Ns, Nd)

        loss_consistency = F.mse_loss(T_ds, T_sd.transpose(1, 2))

        eps = 1e-8
        entropy_ds = -(T_ds * (T_ds + eps).log()).sum(dim=(1, 2)).mean()
        entropy_sd = -(T_sd * (T_sd + eps).log()).sum(dim=(1, 2)).mean()
        loss_entropy = 0.5 * (entropy_ds + entropy_sd)

        # 总 OT 辅助损失
        loss_ot = loss_consistency + 0.01 * loss_entropy
        return loss_ot

def get_loss_function(config):
    if config.loss.lower() == 'infonce':
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        return InfoNCE(loss_function=loss_fn, device=config.device)
