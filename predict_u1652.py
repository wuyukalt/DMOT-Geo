import os
from dataclasses import dataclass
from torch.utils.data import DataLoader

import torch

from models.model import DMOTGeo
from utils.trainer import evaluate
from datasets.university import U1652DatasetEval, get_transforms


@dataclass
class Configuration:
    # Model
    backbone: str = 'ConvNextTiny_2'
    attention: str = 'CMIB'
    aggregation: str = 'DMOT'

    num_channels: int = 384
    img_size: int = 384
    num_clusters: int = 128
    cluster_dim: int = 64

    seed = 1
    verbose: bool = True

    # dmot
    num_scales: int = 3
    sinkhorn_iters: int = 3
    temperature: float = 0.1

    # Eval
    batch_size_eval: int = 32
    eval_gallery_n: int = -1  # -1 for all or int
    normalize_features: bool = True

    # Loss
    loss: str = 'InfoNCE'

    dataset: str = 'U1652-D2S'  # 'U1652-D2S' | 'U1652-S2D'
    data_folder: str = "/University1652-Baseline/data/University-Release"

    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 12
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # for better performance
    cudnn_benchmark: bool = False
    # make cudnn deterministic
    cudnn_deterministic: bool = True
    model_path = r'best_score.pth'


# -----------------------------------------------------------------------------#
# Config                                                                      #
# -----------------------------------------------------------------------------#
# taskset -c 48-85
config = Configuration()

if config.dataset == 'U1652-D2S':
    config.query_folder_test = config.data_folder + '/test/query_drone'
    config.gallery_folder_test = config.data_folder + '/test/gallery_satellite'
elif config.dataset == 'U1652-S2D':
    config.query_folder_test = config.data_folder + '/test/query_satellite'
    config.gallery_folder_test = config.data_folder + '/test/gallery_drone'

if __name__ == '__main__':
    val_transforms, _, _ = get_transforms((config.img_size, config.img_size))
    model = DMOTGeo(config)
    model.load_state_dict(torch.load(config.model_path))
    model = model.to(config.device)

    query_dataset_test = U1652DatasetEval(data_folder=config.query_folder_test, mode="query",
                                          transforms=val_transforms)

    query_dataloader_test = DataLoader(query_dataset_test, batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers, shuffle=False, pin_memory=True)

    gallery_dataset_test = U1652DatasetEval(data_folder=config.gallery_folder_test, mode="gallery",
                                            transforms=val_transforms, sample_ids=query_dataset_test.get_sample_ids(),
                                            gallery_n=config.eval_gallery_n)

    gallery_dataloader_test = DataLoader(gallery_dataset_test, batch_size=config.batch_size_eval,
                                         num_workers=config.num_workers, shuffle=False, pin_memory=True)

    r1_test = evaluate(config=config,
                       model=model,
                       query_loader=query_dataloader_test,
                       gallery_loader=gallery_dataloader_test,
                       ranks=[1, 5, 10],
                       step_size=1000,
                       cleanup=True)

