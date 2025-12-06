import torch
import time
from tqdm import tqdm
from torch.cuda.amp import autocast
import torch.nn.functional as F
import numpy as np
import gc
from utils.util import AverageMeter
import matplotlib.pyplot as plt
from PIL import Image
import os


def train(config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None):
    # set model train mode
    model.train()

    losses = AverageMeter()

    # wait before starting progress bar
    time.sleep(0.1)

    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)

    step = 1

    if config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    # for loop over one epoch
    for query, reference, ids in bar:

        if scaler:
            with autocast():

                # data (batches) to device
                query = query.to(config.device)
                reference = reference.to(config.device)

                # Forward pass
                features1, features2 = model(query, reference)
                # features1(32,8192)
                # features2(32,8192)
                if 'infonce' in config.loss.lower():
                    loss = loss_function(features1, features2, model.logit_scale.exp())
                else:
                    loss = loss_function(features1, features2)
                losses.update(loss.item())

            scaler.scale(loss).backward()

            # Gradient clipping
            if config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), config.clip_grad)

                # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()

            # Scheduler
            if scheduler is not None:
                scheduler.step()

        else:

            # data (batches) to device
            query = query.to(config.device)
            reference = reference.to(config.device)

            # Forward pass
            features1, features2 = model(query, reference)
            loss = loss_function(features1, features2)
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()

            # Gradient clipping
            if config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), config.clip_grad)

                # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()

            # Scheduler
            if scheduler is not None:
                scheduler.step()

        if config.verbose:
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr": "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            bar.set_postfix(ordered_dict=monitor)

        step += 1

    if config.verbose:
        bar.close()

    return losses.avg


def predict(config, model, dataloader):
    model.eval()

    # wait before starting progress bar
    time.sleep(0.1)

    if config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    img_features_list = []

    ids_list = []

    total_infer_time = 0.0

    with torch.no_grad():

        for img, ids in bar:

            ids_list.append(ids)

            with autocast():

                img = img.to(config.device)

                start_time = time.time()

                img_feature = model(img)

                end_time = time.time()

                # normalize is calculated in fp32
                if config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)
            batch_time = end_time - start_time
            total_infer_time += batch_time
            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32))

        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0)
        ids_list = torch.cat(ids_list, dim=0).to(config.device)

    if config.verbose:
        bar.close()

    total_images = len(ids_list)
    print(f"Total inference time: {total_infer_time:.3f} s")
    print(f"Average time per image: {total_infer_time / total_images:.6f} s")

    return img_features, ids_list


def evaluate(config, model, query_loader, gallery_loader, ranks=[1, 5, 10], step_size=1000, cleanup=True):
    print("Extract Features:")
    img_features_query, ids_query = predict(config, model, query_loader)
    # (701,8192), (701,)
    img_features_gallery, ids_gallery = predict(config, model, gallery_loader)
    # (51355,8192), (51355,)

    gl = ids_gallery.cpu().numpy()
    ql = ids_query.cpu().numpy()

    print("Compute Scores:")

    CMC = torch.IntTensor(len(ids_gallery)).zero_()
    ap = 0.0
    for i in tqdm(range(len(ids_query))):
        ap_tmp, CMC_tmp = eval_query(img_features_query[i], ql[i], img_features_gallery, gl)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    AP = ap / len(ids_query) * 100
    CMC = CMC.float()
    CMC = CMC / len(ids_query)  # average CMC
    # top 1%
    top1 = round(len(ids_gallery) * 0.01)
    string = []
    for i in ranks:
        string.append('Recall@{}: {:.4f}'.format(i, CMC[i - 1] * 100))
    string.append('Recall@top1: {:.4f}'.format(CMC[top1] * 100))
    string.append('AP: {:.4f}'.format(AP))

    print(' - '.join(string))
    # cleanup and free memory on GPU
    if cleanup:
        del img_features_query, ids_query, img_features_gallery, ids_gallery
        gc.collect()

    return CMC[0]


def eval_query(qf, ql, gf, gl):
    score = gf @ qf.unsqueeze(-1)
    score = score.squeeze().cpu().numpy()  # [num_gallery, 1] 的向量，每个元素表示 query 与对应 gallery 图像的相似度

    # predict index
    pre_index = np.argsort(score)  # from small to large
    pre_index = pre_index[::-1]  # 反转数组，相似度 从大到小

    # good index
    query_index = np.argwhere(gl == ql)  # 找出 gallery 中哪些图像与当前 query 是同一个目标，ground truth
    good_index = query_index  # 找出query图像对应的真实的gallery图像的index
    # junk index
    junk_index = np.argwhere(gl == -1)  # 有13500个图像是干扰gallery图像，不包含所有query图像对应的gallery图像

    CMC_tmp = compute_mAP(pre_index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(pre_index, good_index, junk_index):
    # pre_index:(51355,) [22 23 36 ...]
    # good_index:(55,1) [[0],[1],[2],...]
    # junk_index:(13500,1) [[109],[110],[111],...]
    ap = 0
    cmc = torch.IntTensor(len(pre_index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(pre_index, junk_index, invert=True)
    pre_index = pre_index[mask]  # (37855,) [22 23 36 ...]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(pre_index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()  # (55,) [0 1 2 3 4 ...]

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc


def visualize_predict(config, model, dataloader):
    model.eval()

    # wait before starting progress bar
    time.sleep(0.1)

    if config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    img_features_list = []

    ids_list = []
    img_paths = []
    with torch.no_grad():

        for img_path, img, ids in bar:

            ids_list.append(ids)

            img_paths.extend(img_path)

            with autocast():

                img = img.to(config.device)
                img_feature = model(img)

                # normalize is calculated in fp32
                if config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)

            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32))

        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0)
        ids_list = torch.cat(ids_list, dim=0).to(config.device)

    if config.verbose:
        bar.close()

    return img_features, ids_list, img_paths


def evaluate_visualize(config, model, query_loader, gallery_loader, ranks=1):
    print("Visualize Predict Results...")
    print("Extract Features:")

    img_features_query, ids_query, img_paths_query = visualize_predict(config, model, query_loader)
    # (701,8192), (701,), list:701
    img_features_gallery, ids_gallery, img_paths_gallery = visualize_predict(config, model, gallery_loader)
    # (51355,8192), (51355,), list:51355

    gallery_label = ids_gallery.cpu().numpy()
    query_label = ids_query.cpu().numpy()

    # 保存文件
    os.makedirs("save_result", exist_ok=True)

    for i in tqdm(range(len(ids_query))):
        # img_features_query[i]
        # query_label[i]
        # img_features_gallery
        # gallery_label
        score = img_features_gallery @ img_features_query[i].unsqueeze(-1)
        score = score.squeeze().cpu().numpy()

        pre_index = np.argsort(score)  # from small to large
        pre_index = pre_index[::-1]  # from large to small

        query_index = np.argwhere(gallery_label == query_label[i])
        good_index = query_index

        print("*" * 30)

        for j in range(ranks):

            if [pre_index[j]] not in good_index:
                pass
            else:
                print("query image:", img_paths_query[i])
                label_str = img_paths_query[i].split("/")[-2]
                img_str = img_paths_query[i].split("/")[-1]
                if "D2S" in config.dataset:
                    label_image_path = "/data1/wuyu/DATA/University-Release/test/gallery_satellite/" + label_str + "/" + label_str + ".jpg"
                elif "S2D" in config.dataset:
                    label_image_path = "/data1/wuyu/DATA/University-Release/test/query_drone/" + label_str + "/" + label_str + ".jpg"
                print("label_image_path:", label_image_path)
                print("result image:", img_paths_gallery[pre_index[j]])
                print(False)

                # 读取图片
                query_img = Image.open(img_paths_query[i]).convert("RGB")
                label_img = Image.open(label_image_path).convert("RGB")
                result_img = Image.open(img_paths_gallery[pre_index[j]]).convert("RGB")

                # 并排画三张
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))

                axes[0].imshow(query_img)
                axes[0].set_title("Query Image")
                axes[0].axis("off")

                axes[1].imshow(label_img)
                axes[1].set_title("Correct Match")
                axes[1].axis("off")

                axes[2].imshow(result_img)
                axes[2].set_title("Retrieved Result")
                axes[2].axis("off")

                plt.tight_layout()


                plt.savefig(f"save_result/query_{label_str}_{img_str}_rank{ranks}.jpg")

        plt.close(fig)
        # plt.show()
        print("*" * 30)
