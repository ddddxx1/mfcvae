"""Render original and reconstructed RGB objects from an NRObjects checkpoint."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from datasets_enrc import Fast_NRObjects
from debug_cmnist_inference import set_final_progressive_state
from load_model import load_model_from_save_dict
from evaluate_nr_objects_cluster_accuracy import checkpoint_factors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default='results/ENRC_nr_objects/reconstructions.png')
    parser.add_argument('--split', choices=('train', 'test'), default='test')
    parser.add_argument('--n_examples', type=int, default=12)
    parser.add_argument('--indices', type=int, nargs='+')
    args = parser.parse_args()
    model, run_args = load_model_from_save_dict(args.model_path, map_location=args.device)
    if run_args.dataset != 'fast_nr_objects' or run_args.model_type != 'conv_vlae':
        raise ValueError('Expected a conv_vlae fast_nr_objects checkpoint')
    names, _ = checkpoint_factors(run_args, model)
    model = model.to(args.device).eval()
    set_final_progressive_state(model, run_args)
    dataset = Fast_NRObjects(root=getattr(run_args, 'nr_objects_root', 'data/enrc_data/nr_objects'),
                         train=args.split == 'train',
                         train_frac=getattr(run_args, 'nr_objects_train_frac', 0.8),
                         seed=getattr(run_args, 'seed', 42), device='cpu',
                         factors_label_list=names)
    indices = args.indices if args.indices is not None else list(range(args.n_examples))
    if not indices or min(indices) < 0 or max(indices) >= len(dataset):
        raise ValueError(f'Indices must be in 0..{len(dataset)-1}')
    inputs = torch.stack([dataset[i][0] for i in indices]).to(args.device)
    labels = torch.stack([dataset[i][1] for i in indices]).tolist()
    with torch.no_grad():
        recons, _, _ = model.forward(inputs, sum(run_args.n_epochs_per_progressive_step)-1, 0)
    inputs = inputs.cpu().permute(0, 2, 3, 1).numpy()
    recons = recons.reshape(-1, 3, 32, 32).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
    fig, axes = plt.subplots(2, len(indices), figsize=(1.8 * len(indices), 4), squeeze=False)
    for col, (index, values) in enumerate(zip(indices, labels)):
        axes[0, col].imshow(inputs[col])
        text = [Fast_NRObjects.LABEL_NAMES[Fast_NRObjects.FACTOR_NAMES.index(name)][value]
                for name, value in zip(names, values)]
        axes[0, col].set_title(f'#{index}\n' + '\n'.join(text), fontsize=8)
        axes[1, col].imshow(recons[col])
        for row in range(2):
            axes[row, col].axis('off')
    axes[0, 0].set_ylabel('input')
    axes[1, 0].set_ylabel('reconstruction')
    fig.tight_layout()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {len(indices)} examples to {path}')


if __name__ == '__main__':
    main()
