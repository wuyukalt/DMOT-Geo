import torch
import torch.nn as nn
import torch.distributed.nn
import torch.nn.functional as F


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


def get_loss_function(config):
    if config.loss.lower() == 'infonce':
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        return InfoNCE(loss_function=loss_fn, device=config.device)
