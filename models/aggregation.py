import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor,
                            iters: int = 20) -> torch.Tensor:
    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)

    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)

    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, dustbin: torch.Tensor, iters: int = 20) -> torch.Tensor:
    B, K, N = scores.shape

    couplings = torch.cat([scores, dustbin], dim=1)  # (B, K+1, N)

    log_mu = torch.full((B, K + 1), -math.log(K + 1), device=scores.device, dtype=scores.dtype)
    log_nu = torch.full((B, N), -math.log(N), device=scores.device, dtype=scores.dtype)

    return log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)


def log_cross_view_ot(feat_q: torch.Tensor, feat_k: torch.Tensor, dustbin: torch.Tensor, temperature: float = 0.1,
                      iters: int = 20) -> torch.Tensor:
    B, C, Nq = feat_q.shape
    _, _, Nk = feat_k.shape

    feat_q = F.normalize(feat_q, p=2, dim=1)
    feat_k = F.normalize(feat_k, p=2, dim=1)

    sim = torch.einsum('bcn,bcm->bnm', feat_q, feat_k) / temperature

    couplings = torch.cat([sim, dustbin], dim=1)  # (B, Nq+1, Nk)

    log_mu = torch.full((B, Nq + 1), -math.log(Nq + 1), device=feat_q.device, dtype=feat_q.dtype)
    log_nu = torch.full((B, Nk), -math.log(Nk), device=feat_q.device, dtype=feat_q.dtype)

    return log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)


class DMOT(nn.Module):
    def __init__(self, in_channels=384, cluster_dim=128, num_clusters=64, num_scales=2, sinkhorn_iters=3, dropout=0.1):
        super().__init__()

        self.in_channels = in_channels
        self.cluster_dim = cluster_dim
        self.num_clusters = num_clusters
        self.num_scales = num_scales
        self.sinkhorn_iters = sinkhorn_iters

        self.feature_mlps = nn.ModuleList()
        self.score_mlps = nn.ModuleList()
        self.dustbin_mlps = nn.ModuleList()

        for _ in range(num_scales):
            self.feature_mlps.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, 512, kernel_size=1),
                    nn.Dropout(dropout),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, cluster_dim, kernel_size=1),
                )
            )
            self.score_mlps.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, 512, kernel_size=1),
                    nn.Dropout(dropout),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, num_clusters, kernel_size=1),
                )
            )
            self.dustbin_mlps.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, 128, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 1, kernel_size=1),
                )
            )

    def _get_scale_feat(self, x, scale_idx):
        if scale_idx == 0:
            return x
        scale = 2 ** scale_idx
        return F.avg_pool2d(x, kernel_size=scale, stride=scale)

    def forward(self, x, return_local=False):
        multi_scale_desc = []
        last_local_feat = None

        for i in range(self.num_scales):
            xs = self._get_scale_feat(x, i)  # (B, C, Hs, Ws)

            local_feat = self.feature_mlps[i](xs).flatten(2)  # (B, C', N)
            score = self.score_mlps[i](xs).flatten(2)  # (B, K, N)
            dustbin = self.dustbin_mlps[i](xs).flatten(2)  # (B, 1, N)

            # 单图 OT
            log_p = log_optimal_transport(score, dustbin, self.sinkhorn_iters)
            p = torch.exp(log_p)  # (B, K+1, N)
            p = p[:, :-1, :]  # 去掉 dustbin -> (B, K, N)

            # cluster-wise 归一化
            p = p / (p.sum(dim=-1, keepdim=True) + 1e-8)

            out = torch.einsum('bcn,bkn->bck', local_feat, p)
            out = F.normalize(out, p=2, dim=1)

            multi_scale_desc.append(out.flatten(1))
            last_local_feat = local_feat

        global_desc = torch.stack(multi_scale_desc, dim=1).mean(dim=1)
        global_desc = F.normalize(global_desc, p=2, dim=1)

        if return_local:
            return global_desc, last_local_feat
        return global_desc


def get_aggregation(aggregation='dmot', in_channels=384, cluster_dim=128, num_clusters=64, num_scales=2,
                    sinkhorn_iters=3):
    if aggregation.lower() == 'dmot':
        return DMOT(
            in_channels=in_channels,
            cluster_dim=cluster_dim,
            num_clusters=num_clusters,
            num_scales=num_scales,
            sinkhorn_iters=sinkhorn_iters
        )
