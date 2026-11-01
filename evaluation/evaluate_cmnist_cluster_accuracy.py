"""评估 C-MNIST 的左右数字聚类准确率，并可视化模型学到的聚类中心。

运行示例：
    python evaluation/evaluate_cmnist_cluster_accuracy.py \
        --model_path wandb/run-.../files/save_dict.pt \
        --device cuda:0 --results_dir /nfs/data8/dong/mfcvae/results/ENRC

Cluster Accuracy 在评估集上通过匈牙利算法，将簇编号与数字标签一对一匹配，
再计算正确样本所占比例。这里没有训练额外的分类器，也不计算 NMI。
聚类中心指模型学习到的 p(z_j | c_j) 的潜变量均值；中心解码图由两个 facet
各取一个中心组合而成，不是数据集中的真实图片。
"""

import argparse
import json
from pathlib import Path

import matplotlib

# 集群任务通常没有图形界面；Agg 后端可以直接把图保存为 PNG。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment    # 求簇与数字之间的一对一最优匹配

from datasets_enrc import Fast_CMNIST
from debug_cmnist_inference import infer, set_final_progressive_state
from load_model import load_model_from_save_dict


def parse_args():
    """读取模型、数据、运行设备以及结果文件命名所需的命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True, help="Path to save_dict.pt")
    parser.add_argument("--results_dir", default="/nfs/data8/dong/mfcvae/results/ENRC")
    parser.add_argument("--output_prefix", default=None,    # 结果文件名前缀
                        help="Prefix for result filenames; defaults to the checkpoint run name")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--data_root", default="data/enrc_data/cmnist")
    return parser.parse_args()


def matched_accuracy(predicted, truth, n_clusters):
    """
    单套clustering准确率
    计算一对一 Cluster Accuracy，并返回簇到数字的映射和诊断数据。
    """
    n_digits = 10
    # counts[c, d]：真实数字为 d、同时被模型分到簇 c 的图片数量。
    counts = np.zeros((n_clusters, n_digits), dtype=np.int64)
    np.add.at(counts, (predicted, truth), 1)    # 遍历所有预测cluster与真实数字，填入计数表
    # 匈牙利算法求最大计数匹配；scipy 默认求最小值，因此传入 -counts。
    # cluster_to_digit 一个簇至多对应一个数字，一个数字也至多对应一个簇。
    rows, cols = linear_sum_assignment(-counts)
    mapping = np.full(n_clusters, -1, dtype=np.int64)
    mapping[rows] = cols
    correct = int(counts[rows, cols].sum()) # 正确的图片数
    return {
        # 此处的匹配和准确率都基于传入的评估集；不是训练集学映射、测试集验证。
        "accuracy": correct / len(truth),
        "correct": correct,
        "cluster_to_digit": mapping.tolist(),
        "contingency_cluster_by_digit": counts.tolist(),
        "cluster_size": counts.sum(axis=1).tolist(),
        # 纯度按每个簇内部最多的真实数字计算，可能与一对一匹配结果不同。
        "cluster_purity": np.divide(
            counts.max(axis=1), counts.sum(axis=1),
            out=np.zeros(n_clusters, dtype=float), where=counts.sum(axis=1) > 0,
        ).tolist(),
    }


def choose_facet_alignment(scores):
    """
    判断哪个facet对应左/右，取总accuracy更高的对齐方式。
    比较两种左右对应方式，保证每个 facet 只分配给一侧数字。
    """
    # scores[facet][label]，其中 label=0 表示左数字，1 表示右数字。
    direct = scores[0][0]["accuracy"] + scores[1][1]["accuracy"]
    swapped = scores[1][0]["accuracy"] + scores[0][1]["accuracy"]
    return (0, 1) if direct >= swapped else (1, 0)


@torch.no_grad()
def decode_center_pairs(model, image_size, batch_size=100):
    """把两个 facet 的中心两两组合并解码；输出按 [facet0, facet1] 索引。"""
    # 每一列是一个簇的潜变量均值，即模型参数中的“聚类中心”。
    centers = [p.detach() for p in model.mu_p_z_j_c_j_list] # 将潜变量均值向量与训练计算图分离
    n0, n1 = (center.shape[1] for center in centers)    # 两个 facet 的簇数
    pairs = [(i, j) for i in range(n0) for j in range(n1)]  # 两个 facet 的簇两两组合，中心配对
    outputs = []
    # 分批解码，避免一次生成过多图片占满显存。
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        z0 = torch.stack([centers[0][:, i] for i, _ in chunk])  # facet0 的潜变量均值（中心向量）
        z1 = torch.stack([centers[1][:, j] for _, j in chunk])
        decoded = model.decode([z0, z1]).detach().cpu() # decoder将中心向量解码为图片
        outputs.append(decoded.reshape(len(chunk), image_size, image_size).numpy())
    return np.concatenate(outputs, axis=0).reshape(n0, n1, image_size, image_size)


def draw_atlas(images, report, results_path):   # 绘制中心图
    """绘制中心组合网格及左右中心侧栏，布局参考用户提供的示意图。"""
    left_facet = report["facet_for_left"]
    right_facet = report["facet_for_right"]
    left_info = report["by_label"]["left"]
    right_info = report["by_label"]["right"]
    # 按一对一匹配得到的数字排序；未匹配的簇（-1）放在最后。
    left_order = sorted(range(len(left_info["cluster_size"])),
                        key=lambda c: (left_info["cluster_to_digit"][c] < 0,
                                       left_info["cluster_to_digit"][c], c))
    right_order = sorted(range(len(right_info["cluster_size"])),
                         key=lambda c: (right_info["cluster_to_digit"][c] < 0,
                                        right_info["cluster_to_digit"][c], c))

    # 解码器始终按 [facet 0, facet 1] 接收中心；若评估后发现左右语义颠倒，
    # 这里需要把“左/右簇编号”换回模型要求的顺序。
    def composition(left_cluster, right_cluster):
        ids = [None, None]
        ids[left_facet] = left_cluster
        ids[right_facet] = right_cluster
        return images[ids[0], ids[1]]

    # 主图：每一行固定一个“左数字”中心，每一列固定一个“右数字”中心。
    mosaic = np.block([[composition(lc, rc) for rc in right_order]
                       for lc in left_order])
    n_left, n_right = len(left_order), len(right_order)
    # 侧栏单独展示一侧中心时，把另一侧固定为样本数最多的簇。
    left_reference = max(range(len(right_info["cluster_size"])),
                         key=lambda c: right_info["cluster_size"][c])
    right_reference = max(range(len(left_info["cluster_size"])),
                          key=lambda c: left_info["cluster_size"][c])

    fig = plt.figure(figsize=(18, 11), facecolor="white")
    gs = fig.add_gridspec(2, 2, width_ratios=[3.3, 1], height_ratios=[1, 1],
                          left=0.06, right=0.98, top=0.90, bottom=0.07,
                          wspace=0.18, hspace=0.24)
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(mosaic, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_facecolor("black")
    height, width = images.shape[-2:]
    ax.set_xticks([(i + 0.5) * width - 0.5 for i in range(n_right)])
    ax.set_xticklabels([f"{right_info['cluster_to_digit'][c]}\n(c{c})"
                        for c in right_order], fontsize=8)
    ax.set_yticks([(i + 0.5) * height - 0.5 for i in range(n_left)])
    ax.set_yticklabels([f"{left_info['cluster_to_digit'][c]} (c{c})"
                        for c in left_order], fontsize=8)
    ax.xaxis.tick_top()
    ax.set_xlabel("Right digit: matched digit (cluster ID)", labelpad=8)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Left digit: matched digit (cluster ID)")
    for i in range(1, n_left):
        ax.axhline(i * height - 0.5, color="white", linestyle=":", linewidth=0.7)
    for i in range(1, n_right):
        ax.axvline(i * width - 0.5, color="white", linestyle=":", linewidth=0.7)

    def draw_center_strip(slot, side, order, fixed, info):
        outer = gs[slot, 1].subgridspec(2, 5, wspace=0.05, hspace=0.30)
        for k, cluster in enumerate(order):
            cell = fig.add_subplot(outer[k // 5, k % 5])
            img = (composition(cluster, fixed) if side == "left"
                   else composition(fixed, cluster))
            cell.imshow(img, cmap="gray", vmin=0, vmax=1)
            cell.set_facecolor("black")
            cell.set_xticks([])
            cell.set_yticks([])
            cell.set_title(f"digit {info['cluster_to_digit'][cluster]} / c{cluster}", fontsize=8)

    # 标准 C-MNIST 每侧 10 个簇，侧栏排为 2×5；其他簇数使用横向长条。
    if n_left == 10 and n_right == 10:
        draw_center_strip(0, "left", left_order, left_reference, left_info)
        draw_center_strip(1, "right", right_order, right_reference, right_info)
        fig.text(0.79, 0.92, f"Left centers | right c{left_reference} fixed",
                 ha="center", fontsize=11)
        fig.text(0.79, 0.49, f"Right centers | left c{right_reference} fixed",
                 ha="center", fontsize=11)
    else:
        for slot, side, order, fixed, info in (
            (0, "left", left_order, left_reference, left_info),
            (1, "right", right_order, right_reference, right_info),
        ):
            panel = fig.add_subplot(gs[slot, 1])
            strip = np.concatenate(
                [composition(c, fixed) if side == "left" else composition(fixed, c)
                 for c in order], axis=1)
            panel.imshow(strip, cmap="gray", vmin=0, vmax=1)
            panel.set_title(f"{side.title()} centers (other facet fixed)")
            panel.axis("off")

    fig.suptitle("C-MNIST: decoded latent cluster centers and their combinations", fontsize=15)
    fig.text(0.06, 0.02,
             "Each tile = decoder(mu_left_cluster, mu_right_cluster). "
             "Side panels hold the opposite facet at its largest cluster. "
             "Digit labels come from evaluation matching.", fontsize=9)
    fig.savefig(results_path, dpi=170, facecolor="white")
    plt.close(fig)


def main():
    """加载 checkpoint，评估左右数字，保存数值报告与中心图。"""
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch_size must be positive")
    model_path = Path(args.model_path).expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    # checkpoint 同时包含模型权重和训练参数，后者用于恢复数据集划分。
    model, run_args = load_model_from_save_dict(str(model_path), map_location=args.device)
    if run_args.dataset != "fast_cmnist" or model.J_n_mixtures != 2:
        raise ValueError("This evaluation requires a two-facet fast_cmnist checkpoint")
    model = model.to(args.device)
    model.device = torch.device(args.device)
    model.eval()
    # 渐进式训练模型必须设置到最后阶段，再直接调用 encode()/decode()。
    set_final_progressive_state(model, run_args)

    # 使用 checkpoint 保存的随机种子和划分比例，重建相同的训练/测试划分。
    dataset = Fast_CMNIST(
        root=args.data_root, train=args.split == "train",
        target_size=getattr(run_args, "cmnist_image_size", 56),
        train_frac=getattr(run_args, "cmnist_train_frac", 5 / 6),
        seed=getattr(run_args, "seed", 42), device="cpu",
    )
    # 每个 facet 分别输出一组簇概率；labels[:, 0/1] 对应左/右真实数字。
    probabilities, labels = infer(model, dataset, run_args.model_type, args.batch_size)
    if labels.ndim != 2 or labels.shape[1] != 2 or len(labels) == 0:
        raise ValueError(f"Expected nonempty [N, 2] labels; got {labels.shape}")
    # 对每张图片、每个 facet，取概率最大的簇作为预测簇号。
    predicted = [p.argmax(axis=1) for p in probabilities]
    # 先计算四种 facet/标签组合，再决定哪个 facet 对应左、哪个对应右。
    scores = [[matched_accuracy(predicted[j], labels[:, label], model.n_clusters_j_list[j])
               for label in range(2)] for j in range(2)]
    left_facet, right_facet = choose_facet_alignment(scores)
    chosen = {"left": scores[left_facet][0], "right": scores[right_facet][1]}

    # JSON 同时保留四种组合的分数、最终匹配、各簇计数及潜变量中心数值。
    report = {
        "checkpoint": str(model_path.resolve()),
        "split": args.split,
        "n_samples": len(labels),
        "metric": "one-to-one Hungarian cluster accuracy, matched on evaluation split",
        "facet_for_left": left_facet,
        "facet_for_right": right_facet,
        "by_label": chosen,
        "all_facet_label_accuracies": {
            f"facet{j}": {"left": scores[j][0]["accuracy"],
                           "right": scores[j][1]["accuracy"]} for j in range(2)
        },
        "latent_cluster_centers": {
            f"facet{j}": model.mu_p_z_j_c_j_list[j].detach().cpu().T.tolist()
            for j in range(2)
        },
    }
    output_dir = Path(args.results_dir).expanduser()
    # 用户已在集群创建结果目录；这里仅检查，不新建目录。
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {output_dir}")
    run_name = model_path.parent.parent.name if model_path.parent.name == "files" else model_path.parent.name
    output_prefix = args.output_prefix or run_name
    if not output_prefix or any(char in output_prefix for char in "/\\"):
        raise ValueError("--output_prefix must be a nonempty filename prefix without path separators")
    # seed0/seed1/seed2 等前缀避免多个任务写入同名结果文件。
    report_path = output_dir / f"{output_prefix}_cluster_accuracy_and_centers.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    atlas_path = output_dir / f"{output_prefix}_cluster_center_atlas.png"
    images = decode_center_pairs(model, getattr(run_args, "cmnist_image_size", 56))
    draw_atlas(images, report, atlas_path)

    print(f"Evaluated {len(labels)} {args.split} images")
    for side, facet in (("left", left_facet), ("right", right_facet)):
        result = chosen[side]
        print(f"{side}: facet {facet}, Cluster Accuracy = {result['accuracy']:.4f} "
              f"({result['correct']}/{len(labels)})")
        print(f"  cluster -> digit: {result['cluster_to_digit']}")
        print(f"  cluster sizes:    {result['cluster_size']}")
    print(f"Center vectors and counts: {report_path}")
    print(f"Decoded center image:      {atlas_path}")


if __name__ == "__main__":
    main()
