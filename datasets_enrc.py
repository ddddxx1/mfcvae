"""
ENRC Benchmark Datasets for MFCVAE
Datasets from 'Deep Embedded Non-Redundant Clustering' (Miklautz et al., AAAI 2020)
Currently supporting:
1. Fast_CMNIST: paired-digit MNIST (60,000 grayscale 56x56 images, 10 left digits x 10 right digits = 100 classes, 2 facets)
2. Fast_Stickfigures: Dancing Stickfigures (900 grayscale images 20x20, 3 facets x 3 classes)
3. Fast_GTSRB: four traffic sign types and two colors (6,720 RGB images 32x32)
4. Fast_NRObjects: single objects with color/material/shape labels (10,000 RGB images resized to 32x32)

将数据集转化为PyTorch可以直接用于训练和评估的格式
用于 MFCVAE 的 ENRC 基准数据集，来自《Deep Embedded Non-Redundant Clustering》（Miklautz 等，AAAI 2020）
当前支持：
1. Fast_CMNIST：双数字 MNIST（60,000 张 56x56 灰度图片，10 个左侧数字 × 10 个右侧数字 = 100 类，2 个方面）
2. Fast_Stickfigures：跳舞火柴人（900 张灰度图 20x20，3 个方面 × 3 类）
"""

import os
import json
from pathlib import Path
import tempfile
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional


class Fast_GTSRB(Dataset):
    """ENRC's four-sign GTSRB subset. Labels are [type, color].

    Class IDs 4/9 are red signs; 35/38 are blue signs. Each track's
    30 adjacent frames stay in the same split to avoid near-duplicate leakage.
    PNG is the supplied contrast-enhanced image; PPM is the original Netpbm
    RGB image. Same-stem files represent one observation, never two.

    数据集按照Track进行划分，否则会出现训练集和测试集包含同一个真实交通标志的照片 - Data Leakage
    模型可能只需要记住训练图片中的视觉特征，就能够很好地处理测试图片。
    """

    TYPE_IDS = (4, 9, 35, 38)
    TYPE_NAMES = ('speed_limit_70', 'no_passing', 'ahead_only', 'keep_right')
    COLOR_NAMES = ('red', 'blue')

    def __init__(self, root='data/enrc_data/gtsrb', train=True,
                 target_size=32, train_frac=0.8, seed=42, device='cpu',
                 force_reprocess=False):
        super().__init__()
        if not 0 < train_frac < 1:
            raise ValueError('train_frac must be between zero and one')
        if target_size != 32:
            raise ValueError('Fast_GTSRB uses 32x32 RGB images for conv_vlae')
        self.device = torch.device(device)
        cache_path = os.path.join(root, 'gtsrb_cache_rgb32_v1.pt')
        if os.path.isfile(cache_path) and not force_reprocess:
            try:
                saved = torch.load(cache_path, map_location='cpu', weights_only=False)
            except TypeError:
                saved = torch.load(cache_path, map_location='cpu')
        else:
            saved = self._preprocess(root, cache_path)
        all_data, all_labels, groups = saved['data'], saved['labels'], saved['groups']
        # Split tracks independently within each sign type, using a local RNG.
        rng = np.random.default_rng(seed)
        selected = np.zeros(len(groups), dtype=bool)
        for type_id in self.TYPE_IDS:   # 4次循环对应4个type_id
            type_groups = sorted({group for group in groups if group[0] == type_id})    # 获取当前类别的所有Track
            perm = rng.permutation(len(type_groups))    # 随机打乱Track的索引
            n_train = max(1, min(len(type_groups) - 1, int(len(type_groups) * train_frac))) # 当前类别应该选择多少个 Track 作为训练集
            train_groups = {type_groups[i] for i in perm[:n_train]} # 选择训练Track
            chosen = train_groups if train else set(type_groups) - train_groups
            selected |= np.array([group in chosen for group in groups])
        indices = np.flatnonzero(selected)
        self.data = all_data[indices].to(self.device)
        self.labels = all_labels[indices].to(self.device)
        self.groups = [groups[i] for i in indices]
        print(f'[Fast_GTSRB] {"Train" if train else "Test"}: {len(self.data)} images on {self.device}')

    # 读取原始图片，转换为tensor，并保存缓存
    @classmethod
    def _preprocess(cls, root, cache_path):
        from PIL import Image

        entries = []
        for type_index, type_id in enumerate(cls.TYPE_IDS):
            folder = os.path.join(root, 'train', f'{type_id:05d}')
            if not os.path.isdir(folder):
                raise FileNotFoundError(f'Missing GTSRB class folder: {folder}')
            paths = {}
            for name in sorted(os.listdir(folder)):
                stem, ext = os.path.splitext(name)
                if ext.lower() in ('.png', '.ppm'):
                    # Prefer the supplied enhanced PNG; retain PPM support.
                    if stem not in paths or ext.lower() == '.png':
                        paths[stem] = os.path.join(folder, name)
            for stem, path in sorted(paths.items()):
                track, frame = stem.split('_')  # dataset: track_frame.png
                if not (track.isdigit() and frame.isdigit()):
                    raise ValueError(f'Unexpected GTSRB filename: {path}')
                entries.append((path, type_index, 0 if type_id in (4, 9) else 1,
                                (type_id, int(track))))
        if not entries:
            raise RuntimeError(f'No GTSRB images in {root}')
        data = torch.empty((len(entries), 3, 32, 32), dtype=torch.float32)
        labels = torch.empty((len(entries), 2), dtype=torch.long)
        groups = []
        for i, (path, type_label, color_label, group) in enumerate(entries):
            with Image.open(path) as image:
                image = image.convert('RGB')
                if image.size != (32, 32):
                    image = image.resize((32, 32), Image.BILINEAR)
                data[i] = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy()) / 255.0
            labels[i] = torch.tensor((type_label, color_label))
            groups.append(group)
        saved = {'data': data, 'labels': labels, 'groups': groups}
        torch.save(saved, cache_path)
        print(f'[Fast_GTSRB] Cached {len(entries)} distinct RGB images to {cache_path}')
        return saved

    # 返回数据集中的图片数量
    def __len__(self):
        return len(self.data)

    # 根据索引返回图片和标签
    def __getitem__(self, index):
        return self.data[index], self.labels[index]


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
                 target_size: int = 56,
                 train_frac: float = 5/6,  # 50,000 train, 10,000 test
                 seed: int = 42,
                 device: str = 'cpu',
                 force_reprocess: bool = False):
        super().__init__()
        self.root = root
        self.target_size = target_size
        self.device = torch.device(device)

        # 缓存机制，节约时间
        # 灰度缓存与传统的 RGB 缓存不同。
        cache_path = os.path.join(
            os.path.dirname(root), f"cmnist_cache_gray_{target_size}x{target_size}.pt")

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
        data_tensor = torch.zeros((total_images, 1, target_size, target_size), dtype=torch.float32)
        labels_tensor = torch.zeros((total_images, 2), dtype=torch.long)

        for i, (path, left_digit_label, right_digit_label) in enumerate(image_entries):
            # ENRC C-MNIST 是一个单通道灰度数据集。
            img = Image.open(path).convert('L')
            if img.size != (target_size, target_size):  # 调整原始图片尺寸
                img = img.resize((target_size, target_size), Image.BILINEAR)    # 双线性插值
            arr = np.array(img, dtype=np.float32)[None, ...] / 255.0  # (1, H, W)
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


class Fast_NRObjects(Dataset):
    """ENRC single-object renders, with labels [color, material, shape].

    Read individual scenes, not the incomplete root aggregate JSON. Images
    are resized from 128x128 to the existing conv_vlae RGB input size (32x32).
    The deterministic split is stratified by all three attributes together.

    NRC 单对象渲染，带有标签 [颜色、材质、形状]。
    读取单个场景，而不是不完整的根聚合 JSON。图像从 128x128 调整为现有的 conv_vlae RGB 输入尺寸（32x32）。确定性拆分是基于三种属性一起进行分层的。
    """

    FACTOR_NAMES = ('color', 'material', 'shape')
    COLOR_NAMES = ('blue', 'gray', 'green', 'purple', 'red', 'yellow')
    MATERIAL_NAMES = ('metal', 'rubber')
    SHAPE_NAMES = ('cube', 'cylinder', 'sphere')
    LABEL_NAMES = (COLOR_NAMES, MATERIAL_NAMES, SHAPE_NAMES)

    def __init__(self,
                 root='data/enrc_data/nr_objects',
                 train=True,
                 target_size=32,
                 train_frac=0.8,
                 seed=42,
                 device='cpu',
                 force_reprocess=False,
                 factors_label_list=None):
        super().__init__()
        if not 0 < train_frac < 1:
            raise ValueError('train_frac must be between zero and one')
        if target_size != 32:
            raise ValueError('Fast_NRObjects uses 32x32 RGB images')
        self.factors_label_list = list(factors_label_list or self.FACTOR_NAMES)
        if (not self.factors_label_list or len(set(self.factors_label_list)) != len(self.factors_label_list)
                or any(name not in self.FACTOR_NAMES for name in self.factors_label_list)):
            raise ValueError('Select distinct NRObjects factors: color, material, shape')
        self.device = torch.device(device)
        root = Path(root)
        # 优先加载缓存，首次运行才处理图片
        cache_path = root / 'nr_objects_cache_rgb32_v1.pt'
        if cache_path.is_file() and not force_reprocess:
            try:
                saved = torch.load(cache_path, map_location='cpu', weights_only=False)
            except TypeError:
                saved = torch.load(cache_path, map_location='cpu')
        else:
            saved = self._preprocess(root)
            # Array jobs may start together: publish only a complete cache.
            with tempfile.NamedTemporaryFile(dir=root, suffix='.pt', delete=False) as f:
                temporary = f.name
            try:
                torch.save(saved, temporary)
                os.replace(temporary, cache_path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        rng = np.random.default_rng(seed)
        labels = saved['labels'].numpy()
        selected = []
        for combination in np.unique(labels, axis=0):
            indices = rng.permutation(np.flatnonzero((labels == combination).all(axis=1)))
            if len(indices) < 2:
                raise ValueError(f'Need at least two images for combination {combination}')
            n_train = max(1, min(len(indices) - 1, int(len(indices) * train_frac)))
            selected.extend(indices[:n_train] if train else indices[n_train:])
        self.indices = np.sort(selected)
        self.data = saved['data'][self.indices].to(self.device)
        columns = [self.FACTOR_NAMES.index(name) for name in self.factors_label_list]
        self.labels = saved['labels'][self.indices][:, columns].to(self.device)
        self.image_filenames = [saved['image_filenames'][i] for i in self.indices]
        print(f'[Fast_NRObjects] {"Train" if train else "Test"}: {len(self.data)} images on {self.device}')

    @classmethod
    def _preprocess(cls, root):
        from PIL import Image

        paths = sorted((root / 'images' / 'train').glob('*.png'))
        if not paths:
            raise FileNotFoundError(f'No images found in {root / "images" / "train"}')
        data = torch.empty((len(paths), 3, 32, 32), dtype=torch.float32)
        labels = torch.empty((len(paths), 3), dtype=torch.long)
        # 逐张读取图片对应的json
        for i, path in enumerate(paths):
            scene_path = root / 'scenes' / (path.stem + '.json')
            with scene_path.open() as f:
                scene = json.load(f)
            if scene['image_filename'] != path.name or len(scene['objects']) != 1:
                raise ValueError(f'Expected matching single-object scene: {scene_path}')
            obj = scene['objects'][0]
            labels[i] = torch.tensor([names.index(obj[factor])
                                      for factor, names in zip(cls.FACTOR_NAMES, cls.LABEL_NAMES)])
            with Image.open(path) as image:
                image = image.convert('RGB').resize((32, 32), Image.BILINEAR)
                data[i] = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy()) / 255.0
        return {'data': data, 'labels': labels, 'image_filenames': [p.name for p in paths]}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
