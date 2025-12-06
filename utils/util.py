import os
import sys
import random
import errno
import time
import torch
import numpy as np
from datetime import timedelta
from shutil import copyfile, copytree, rmtree


class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count


def setup_system(seed, cudnn_benchmark=True, cudnn_deterministic=True) -> None:
    '''
    Set seeds for for reproducible training
    '''
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)  # 固定 Python 内置 random 库的随机数种子
    np.random.seed(seed)  # 固定 NumPy 的随机数种子
    torch.manual_seed(seed)  # 设置 CPU 随机种子
    torch.cuda.manual_seed(seed)  # 设置 GPU 随机种子
    torch.cuda.manual_seed_all(seed)  # 如果使用多 GPU，设置所有 GPU 的随机种子
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = cudnn_benchmark  # 禁用自动优化，以保证结果的稳定性
        torch.backends.cudnn.deterministic = cudnn_deterministic  # 确保每次运行相同输入时，卷积等操作的一致性


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def sec_to_min(seconds):
    seconds = int(seconds)
    minutes = seconds // 60
    seconds_remaining = seconds % 60

    if seconds_remaining < 10:
        seconds_remaining = '0{}'.format(seconds_remaining)

    return '{}:{}'.format(minutes, seconds_remaining)


def sec_to_time(seconds):
    return "{:0>8}".format(str(timedelta(seconds=int(seconds))))


def print_time_stats(t_train_start, t_epoch_start, epochs_remaining, steps_per_epoch):
    elapsed_time = time.time() - t_train_start
    speed_epoch = time.time() - t_epoch_start
    speed_batch = speed_epoch / steps_per_epoch
    eta = speed_epoch * epochs_remaining

    print("Elapsed {}, {} time/epoch, {:.2f} s/batch, remaining {}".format(
        sec_to_time(elapsed_time), sec_to_time(speed_epoch), speed_batch, sec_to_time(eta)))


def copy_file_or_tree(path, target_dir):
    target_path = os.path.join(target_dir, path)
    if os.path.isdir(path):
        if os.path.exists(target_path):
            rmtree(target_path)
        copytree(path, target_path)
    elif os.path.isfile(path):
        copyfile(path, target_path)


def copyfiles2checkpoints(target_path, datatype):
    if not os.path.isdir(target_path):
        os.mkdir(target_path)
    # record every run
    if datatype == "u1652":
        copy_file_or_tree('predict_u1652.py', target_path)
        copy_file_or_tree('visualize_predict_u1652.py', target_path)
    elif datatype == "s200":
        copy_file_or_tree('predict_s200.py', target_path)
        copy_file_or_tree('visualize_predict_s200.py', target_path)
    copy_file_or_tree('datasets', target_path)
    copy_file_or_tree('models', target_path)
    copy_file_or_tree('utils', target_path)
