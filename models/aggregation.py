import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


# Code from SuperGlue (https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py)
def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ 在对数空间中执行 Sinkhorn 归一化以确保稳定性"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ 在对数空间中执行可微分最优传输以实现稳定性"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns, bs = (m * one).to(scores), (n * one).to(scores), ((n - m) * one).to(scores)

    bins = alpha.expand(b, 1, n)

    couplings = torch.cat([scores, bins], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), bs.log()[None] + norm])
    log_nu = norm.expand(n)
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


class OFTA(nn.Module):
    def __init__(self, num_channels=384, num_clusters=64, cluster_dim=128, dropout=0.3):
        super(OFTA, self).__init__()
        self.num_channels = num_channels
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim

        dropout = nn.Dropout(dropout)
        # MLP for local features f_i
        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, 1)
        )
        # MLP for score matrix S
        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.num_clusters, 1),
        )
        # Dustbin parameter z
        self.dust_bin = nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        # 经过注意力之后的特征 (32,384,24,24)
        f = self.cluster_features(x).flatten(2)  # (32,64,24,24)->(32,64,24*24=576)
        p = self.score(x).flatten(2)  # (32,128,24,24)->(32,128,24*24=576)

        # Sinkhorn algorithm
        p = log_optimal_transport(p, self.dust_bin, 3)  # (32,129,576)
        p = torch.exp(p)
        # Normalize to maintain mass
        p = p[:, :-1, :]  # (32,128,576)

        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)  # (32,64,128,576)
        f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)  # (32,64,128,576)
        out = F.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1)  # (32,64,128,576)->(32,64,128)->(32,64*128=8192)

        return out


def get_aggregation(aggregation='ofta', num_channels=384, num_clusters=64, cluster_dim=128):
    if aggregation.lower() == 'ofta':
        return OFTA(num_channels=num_channels, num_clusters=num_clusters, cluster_dim=cluster_dim)
