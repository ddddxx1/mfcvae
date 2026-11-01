"""Inspect C-MNIST labels and per-facet cluster predictions from a checkpoint.

Example:
  python debug_cmnist_inference.py --model_path outputs/.../save_dict.pt --device cuda:0

Cluster IDs are arbitrary.  This script first learns a *separate* majority-vote
cluster-to-digit mapping for each (facet, label) from the training split, then
prints predictions for a few test images whose two digits differ.

检查 C-MNIST 标签以及来自检查点的每个方面的聚类预测。
例如：python debug_cmnist_inference.py --model_path outputs...save_dict.pt --device cuda:0
聚类 ID 是任意的。该脚本首先从训练集为每个（facet, label）学习一个单独的多数投票聚类到数字的映射，然后打印一些测试图片的预测，这些图片的两个数字不同。
"""

import argparse

import numpy as np
import torch

from datasets_enrc import Fast_CMNIST
from load_model import load_model_from_save_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--n_examples", type=int, default=8)
    return parser.parse_args()


def set_final_progressive_state(model, run_args):
    """Match the final training stage before calling model.encode()."""
    if not getattr(model, "do_progressive_training", False):
        return
    final_epoch = sum(run_args.n_epochs_per_progressive_step) - 1
    alpha_enc, alpha_dec, gamma_z, gamma_c = model.compute_progressive_training_coefficients(
        final_epoch, 0
    )
    model.alpha_enc_fade_in_list = alpha_enc
    model.alpha_dec_fade_in_list = alpha_dec
    model.gamma_kl_z_list = gamma_z
    model.gamma_kl_c_list = gamma_c
    model.encoder.alpha_enc_fade_in_list = alpha_enc
    model.decoder.alpha_dec_fade_in_list = alpha_dec


@torch.no_grad()
def infer(model, dataset, model_type, batch_size):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probabilities = [[] for _ in range(model.J_n_mixtures)]
    labels = []
    for x, y in loader:
        if model_type in ("fc_shared", "fc_per_facet_enc_shared_dec", "fc_vlae"):
            x = x.view(x.size(0), -1).float()
        else:
            x = x.float()
        x = x.to(model.device)
        mu_list, _ = model.encode(x)
        q_c_list = model.compute_q_c_j_x(mu_list)
        for facet, q_c in enumerate(q_c_list):
            probabilities[facet].append(q_c.cpu().numpy())
        labels.append(y.cpu().numpy())
    return [np.concatenate(p, axis=0) for p in probabilities], np.concatenate(labels, axis=0)


def majority_mapping(clusters, labels, n_clusters):
    """Return cluster -> most common digit; no cross-facet mapping is reused."""
    mapping = np.full(n_clusters, -1, dtype=int)
    table = np.zeros((n_clusters, 10), dtype=int)
    for cluster, label in zip(clusters, labels):
        table[cluster, label] += 1
    nonempty = table.sum(axis=1) > 0
    mapping[nonempty] = table[nonempty].argmax(axis=1)
    purity = np.divide(
        table.max(axis=1), table.sum(axis=1), out=np.zeros(n_clusters), where=nonempty
    )
    return mapping, table, purity


def main():
    args = parse_args()
    model, run_args = load_model_from_save_dict(args.model_path, map_location=args.device)
    if run_args.dataset != "fast_cmnist":
        raise ValueError(f"Expected a fast_cmnist checkpoint, got {run_args.dataset!r}.")
    model = model.to(args.device)
    model.eval()
    set_final_progressive_state(model, run_args)

    size = getattr(run_args, "cmnist_image_size", 56)
    frac = getattr(run_args, "cmnist_train_frac", 5 / 6)
    seed = getattr(run_args, "seed", 42)
    train = Fast_CMNIST(train=True, target_size=size, train_frac=frac, seed=seed, device="cpu")
    test = Fast_CMNIST(train=False, target_size=size, train_frac=frac, seed=seed, device="cpu")

    # Dataset-level guard against the suspected label-column bug.
    for name, dataset in (("train", train), ("test", test)):
        labels = dataset.labels.cpu().numpy()
        print(
            f"{name}: data={tuple(dataset.data.shape)}, labels={tuple(labels.shape)}, "
            f"same-label rate={(labels[:, 0] == labels[:, 1]).mean():.4f}, "
            f"unique pairs={len(np.unique(labels, axis=0))}"
        )

    train_probs, train_labels = infer(model, train, run_args.model_type, args.batch_size)
    test_probs, test_labels = infer(model, test, run_args.model_type, args.batch_size)
    train_clusters = [p.argmax(axis=1) for p in train_probs]
    test_clusters = [p.argmax(axis=1) for p in test_probs]

    mappings = {}
    names = ("left", "right")
    for facet in range(model.J_n_mixtures):
        print(f"\nFacet {facet} cluster diagnostics:")
        for label_col, name in enumerate(names):
            mapping, table, purity = majority_mapping(
                train_clusters[facet], train_labels[:, label_col], model.n_clusters_j_list[facet]
            )
            prediction = mapping[test_clusters[facet]]
            accuracy = (prediction == test_labels[:, label_col]).mean()
            print(
                f"  against {name:5s}: mapped test accuracy={accuracy:.4f}; "
                f"cluster->digit={mapping.tolist()}; purity={np.round(purity, 3).tolist()}"
            )
            mappings[facet, label_col] = mapping

    # Prefer examples proving that the two labels really differ.
    differing = np.flatnonzero(test_labels[:, 0] != test_labels[:, 1])
    chosen = differing[: args.n_examples]
    print("\nSample inference (cluster IDs are arbitrary; mapped digits use the TRAIN split):")
    for index in chosen:
        items = [f"index={index}", f"ground_truth=(left={test_labels[index, 0]}, right={test_labels[index, 1]})"]
        for facet in range(model.J_n_mixtures):
            cluster = test_clusters[facet][index]
            confidence = test_probs[facet][index, cluster]
            left = mappings[facet, 0][cluster]
            right = mappings[facet, 1][cluster]
            items.append(
                f"facet{facet}: cluster={cluster}, q={confidence:.3f}, maps_to=(left={left}, right={right})"
            )
        print(" | ".join(items))


if __name__ == "__main__":
    main()
