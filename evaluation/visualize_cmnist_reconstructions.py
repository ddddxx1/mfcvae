"""Render input/reconstruction pairs from a trained C-MNIST checkpoint.

Example:
  python visualize_cmnist_reconstructions.py \
    --model_path outputs/disabled_seed0/save_dict.pt \
    --device cuda:0 \
    --output results/cmnist_reconstructions_seed0.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import torch

from datasets_enrc import Fast_CMNIST
from load_model import load_model_from_save_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="results/cmnist_reconstructions.png")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--n_examples", type=int, default=12)
    parser.add_argument(
        "--indices", type=int, nargs="+",
        help="Optional dataset indices; otherwise use the first n_examples.",
    )
    return parser.parse_args()


def set_final_progressive_state(model, run_args):
    """Use the same fully-faded-in state as at the final training epoch."""
    if not getattr(model, "do_progressive_training", False):
        return
    epoch = sum(run_args.n_epochs_per_progressive_step) - 1
    alpha_enc, alpha_dec, gamma_z, gamma_c = model.compute_progressive_training_coefficients(epoch, 0)
    model.alpha_enc_fade_in_list = alpha_enc
    model.alpha_dec_fade_in_list = alpha_dec
    model.gamma_kl_z_list = gamma_z
    model.gamma_kl_c_list = gamma_c
    model.encoder.alpha_enc_fade_in_list = alpha_enc
    model.decoder.alpha_dec_fade_in_list = alpha_dec


def main():
    cli = parse_args()
    model, run_args = load_model_from_save_dict(cli.model_path, map_location=cli.device)
    if run_args.dataset != "fast_cmnist":
        raise ValueError(f"Expected fast_cmnist checkpoint, got {run_args.dataset!r}.")
    if run_args.model_type != "conv_vlae":
        raise ValueError(f"This script currently expects conv_vlae, got {run_args.model_type!r}.")

    model = model.to(cli.device).eval()
    set_final_progressive_state(model, run_args)
    size = getattr(run_args, "cmnist_image_size", 56)
    dataset = Fast_CMNIST(
        train=cli.split == "train",
        target_size=size,
        train_frac=getattr(run_args, "cmnist_train_frac", 5 / 6),
        seed=getattr(run_args, "seed", 42),
        device="cpu",
    )
    indices = cli.indices or list(range(cli.n_examples))
    if not indices or min(indices) < 0 or max(indices) >= len(dataset):
        raise ValueError(f"--indices must be between 0 and {len(dataset) - 1}.")

    inputs = torch.stack([dataset[i][0] for i in indices]).to(cli.device).float()
    labels = torch.stack([dataset[i][1] for i in indices]).cpu().numpy()
    with torch.no_grad():
        reconstructions, _, _ = model.forward(
            inputs, sum(run_args.n_epochs_per_progressive_step) - 1, 0
        )
    inputs = inputs.cpu().numpy()
    reconstructions = reconstructions.clamp(0, 1).cpu().numpy()

    fig, axes = plt.subplots(2, len(indices), figsize=(1.8 * len(indices), 3.8), squeeze=False)
    for column, (index, label) in enumerate(zip(indices, labels)):
        axes[0, column].imshow(inputs[column, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, column].set_title(f"#{index}\nL={label[0]}, R={label[1]}", fontsize=9)
        axes[1, column].imshow(reconstructions[column, 0], cmap="gray", vmin=0, vmax=1)
        for row in range(2):
            axes[row, column].axis("off")
    axes[0, 0].set_ylabel("input", fontsize=11)
    axes[1, 0].set_ylabel("reconstruction", fontsize=11)
    fig.suptitle("C-MNIST autoencoder output", y=0.99)
    fig.tight_layout()

    output_dir = os.path.dirname(cli.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(cli.output, dpi=180, bbox_inches="tight")
    print(f"Saved {len(indices)} input/reconstruction pairs to {cli.output}")


if __name__ == "__main__":
    main()
