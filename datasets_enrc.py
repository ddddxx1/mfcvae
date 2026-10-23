"""
ENRC Benchmark Datasets for MFCVAE
Datasets from 'Deep Embedded Non-Redundant Clustering' (Miklautz et al., AAAI 2020)
Currently supporting:
1. Fast_CMNIST: paired-digit MNIST (60,000 RGB images, 10 left digits x 10 right digits = 100 classes, 2 facets)
2. Fast_Stickfigures: Dancing Stickfigures (900 grayscale images 20x20, 3 facets x 3 classes)

将数据集转化为PyTorch可以直接用于训练和评估的格式
用于 MFCVAE 的 ENRC 基准数据集，来自《Deep Embedded Non-Redundant Clustering》（Miklautz 等，AAAI 2020）
当前支持：
1. Fast_CMNIST：双数字 MNIST（60,000 张 RGB 图片，10 个左侧数字 × 10 个右侧数字 = 100 类，2 个方面）
2. Fast_Stickfigures：跳舞火柴人（900 张灰度图 20x20，3 个方面 × 3 类）
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional


class Fast_CMNIST(Dataset):
    """
    Paired-digit MNIST (CMNIST) dataset from the ENRC benchmark.
    Contains 60,000 images structured in 100 folders (0_0 to 9_9).
    Folder name format: '<left_digit>_<right_digit>', with both labels in [0..9].
    Multi-ground-truth facets:
      - Facet 0: left-digit label (0..9)
      - Facet 1: right-digit label (0..9)

    来自 ENRC 基准的双数字 MNIST (CMNIST) 数据集。
    包含 60,000 张图像，分布在 100 个文件夹中（0_0 到 9_9）。
    文件夹名称格式：'<left_digit>_<right_digit>'，两个数字均在 [0..9]。
    多重真实标签维度：
        - 维度 0：左侧数字标签 (0..9)
        - 维度 1：右侧数字标签 (0..9)
    """
    def __init__(self,
                 root: str = 'data/enrc_data/cmnist',
                 train: bool = True,
                 target_size: int = 32,
                 train_frac: float = 5/6,  # 50,000 train, 10,000 test
                 seed: int = 42,
                 device: str = 'cpu',
                 force_reprocess: bool = False):
        super().__init__()
        self.root = root
        self.target_size = target_size
        self.device = torch.device(device)

        # 缓存机制，节约时间
        cache_path = os.path.join(os.path.dirname(root), f"cmnist_cache_{target_size}x{target_size}.pt")

        if os.path.exists(cache_path) and not force_reprocess:
            print(f"[Fast_CMNIST] Loading preprocessed data from cache: {cache_path}")
            try:
                saved = torch.load(cache_path, map_location='cpu', weights_only=False)
            except TypeError:
                saved = torch.load(cache_path, map_location='cpu')
            all_data = saved['data']
            all_labels = saved['labels']
        else:   # 缓存不存在
            print(f"[Fast_CMNIST] Preprocessing 60,000 images from {root} (target_size={target_size}x{target_size})...")
            all_data, all_labels = self._preprocess_and_cache(root, cache_path, target_size)

        # Train / Test split
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_total = len(all_data)
        perm = np.random.permutation(n_total)
        n_train = int(n_total * train_frac)

        if train:
            indices = perm[:n_train]
        else:
            indices = perm[n_train:]

        self.data = all_data[indices].to(self.device)
        self.labels = all_labels[indices].to(self.device)
        split_name = "Train" if train else "Test"
        print(f"[Fast_CMNIST] {split_name} set initialized: {len(self.data)} samples on device {self.device}")

    # 数据加载流程
    def _preprocess_and_cache(self, root: str, cache_path: str, target_size: int):
        from PIL import Image

        folder_names = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and not d.startswith('.')])
        if len(folder_names) == 0:
            raise RuntimeError(f"No class folders found in {root}!")

        image_entries = []  # 文件夹中的所有图片label
        for folder in folder_names:
            parts = folder.split('_')
            if len(parts) != 2:
                continue
            left_digit_label = int(parts[0])    # 文件夹名left digit
            right_digit_label = int(parts[1])   # 文件夹名right digit
            folder_path = os.path.join(root, folder)
            for fname in sorted(os.listdir(folder_path)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_entries.append((os.path.join(folder_path, fname), left_digit_label, right_digit_label))

        total_images = len(image_entries)
        print(f"[Fast_CMNIST] Found {total_images} images across {len(folder_names)} folders.")

        # 图片和labels存储成tensor
        data_tensor = torch.zeros((total_images, 3, target_size, target_size), dtype=torch.float32)
        labels_tensor = torch.zeros((total_images, 2), dtype=torch.long)

        for i, (path, left_digit_label, right_digit_label) in enumerate(image_entries):
            img = Image.open(path).convert('RGB')   # RGB格式，所有图片都有三个通道
            if img.size != (target_size, target_size):  # 调整原始图片尺寸
                img = img.resize((target_size, target_size), Image.BILINEAR)    # 双线性插值
            arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, C) 归一化
            arr = np.transpose(arr, (2, 0, 1))             # (C, H, W)  PyTorch卷积神经网络格式
            data_tensor[i] = torch.from_numpy(arr)
            labels_tensor[i, 0] = left_digit_label  # facet0
            labels_tensor[i, 1] = right_digit_label # facet1

            if (i + 1) % 10000 == 0 or (i + 1) == total_images:
                print(f"[Fast_CMNIST] Processed {i + 1}/{total_images} images...")

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        print(f"[Fast_CMNIST] Saving cache to {cache_path}...")
        torch.save({'data': data_tensor, 'labels': labels_tensor}, cache_path)
        return data_tensor, labels_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


class Fast_Stickfigures(Dataset):
    """
    Dancing Stickfigures dataset from ENRC benchmark.
    Contains 900 synthetic grayscale images (20x20) in stickfigures_3sub.data.
    Each sample has:
      - 3 ground-truth labels (posture attributes), each in [0, 1, 2]
      - 400 grayscale pixels (values 0..255)
    """
    def __init__(self,
                 data_file: str = 'data/enrc_data/stickfigures/stickfigures_3sub.data',
                 train: bool = True,
                 train_frac: float = 0.8,
                 seed: int = 42,
                 device: str = 'cpu'):
        super().__init__()
        self.device = torch.device(device)

        cache_path = os.path.join(os.path.dirname(data_file), "stickfigures_cache.pt")
        if os.path.exists(cache_path):
            print(f"[Fast_Stickfigures] Loading preprocessed data from cache: {cache_path}")
            try:
                saved = torch.load(cache_path, map_location='cpu', weights_only=False)
            except TypeError:
                saved = torch.load(cache_path, map_location='cpu')
            all_data = saved['data']
            all_labels = saved['labels']
        else:
            print(f"[Fast_Stickfigures] Parsing {data_file}...")
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"Stickfigures data file not found at {data_file}!")

            labels_list = []
            pixels_list = []
            with open(data_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(';')
                    # first 3 values are labels
                    lbls = [int(parts[0]), int(parts[1]), int(parts[2])]
                    pxls = [float(p) for p in parts[3:403]]
                    labels_list.append(lbls)
                    pixels_list.append(pxls)

            all_labels = torch.tensor(labels_list, dtype=torch.long)
            all_data = torch.tensor(pixels_list, dtype=torch.float32).view(-1, 1, 20, 20) / 255.0
            torch.save({'data': all_data, 'labels': all_labels}, cache_path)
            print(f"[Fast_Stickfigures] Cached {len(all_data)} samples to {cache_path}.")

        # Train / Test split
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_total = len(all_data)
        perm = np.random.permutation(n_total)
        n_train = int(n_total * train_frac)

        if train:
            indices = perm[:n_train]
        else:
            indices = perm[n_train:]

        self.data = all_data[indices].to(self.device)
        self.labels = all_labels[indices].to(self.device)
        split_name = "Train" if train else "Test"
        print(f"[Fast_Stickfigures] {split_name} set initialized: {len(self.data)} samples on device {self.device}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
