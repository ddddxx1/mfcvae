"""Evaluate ENRC NRObjects color/material or color/shape clustering and decode RGB cluster centers.

Run from repository root with PYTHONPATH=$PWD:$PWD/evaluation.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from datasets_enrc import Fast_NRObjects
from debug_cmnist_inference import infer, set_final_progressive_state
from load_model import load_model_from_save_dict


def matched_accuracy(predicted, truth, n_clusters, n_classes):
    counts = np.zeros((n_clusters, n_classes), dtype=np.int64)
    np.add.at(counts, (predicted, truth), 1)
    rows, cols = linear_sum_assignment(-counts)
    mapping = np.full(n_clusters, -1, dtype=np.int64)
    mapping[rows] = cols
    correct = int(counts[rows, cols].sum())
    return {
        'accuracy': correct / len(truth), 'correct': correct,
        'cluster_to_label': mapping.tolist(),
        'contingency_cluster_by_label': counts.tolist(),
        'cluster_size': counts.sum(axis=1).tolist(),
        'cluster_purity': np.divide(counts.max(axis=1), counts.sum(axis=1),
                                    out=np.zeros(n_clusters),
                                    where=counts.sum(axis=1) > 0).tolist(),
    }


def checkpoint_factors(run_args, model, experiment=None):
    names = list(run_args.factors_label_list)
    if names not in (['color', 'material'], ['color', 'shape']):
        raise ValueError('Expected color/material or color/shape checkpoint')
    if experiment is not None and names != ['color', 'material' if experiment == 1 else 'shape']:
        raise ValueError('Checkpoint does not match the requested experiment')
    sizes = [len(Fast_NRObjects.LABEL_NAMES[Fast_NRObjects.FACTOR_NAMES.index(n)]) for n in names]
    if model.J_n_mixtures != 2 or sorted(model.n_clusters_j_list) != sorted(sizes):
        raise ValueError(f'Expected two facets with cluster counts {sizes}')
    return names, sizes


@torch.no_grad()
def draw_atlas(model, names, facets, chosen, output_path):
    """Rows are colors; columns are material or shape centers."""
    label_names = {name: Fast_NRObjects.LABEL_NAMES[Fast_NRObjects.FACTOR_NAMES.index(name)]
                   for name in names}
    orders = {name: sorted(range(len(label_names[name])),
                          key=lambda c: chosen[name]['cluster_to_label'][c]) for name in names}
    tiles = []
    for color in orders[names[0]]:
        row = []
        for other in orders[names[1]]:
            ids = [None, None]
            ids[facets[names[0]]], ids[facets[names[1]]] = color, other
            z = [p[:, ids[j]].unsqueeze(0) for j, p in enumerate(model.mu_p_z_j_c_j_list)]
            row.append(model.decode(z).reshape(3, 32, 32).clamp(0, 1).cpu().permute(1, 2, 0).numpy())
        tiles.append(np.concatenate(row, axis=1))
    fig, ax = plt.subplots(figsize=(7, 11))
    ax.imshow(np.concatenate(tiles, axis=0))
    ax.set_xticks([16 + 32 * i for i in range(len(orders[names[1]]))])
    ax.set_xticklabels([label_names[names[1]][chosen[names[1]]['cluster_to_label'][c]]
                       + f' (c{c})' for c in orders[names[1]]])
    ax.set_yticks([16 + 32 * i for i in range(6)])
    ax.set_yticklabels([label_names['color'][chosen['color']['cluster_to_label'][c]]
                       + f' (c{c})' for c in orders['color']])
    ax.set_xlabel(names[1])
    ax.set_ylabel('color')
    ax.set_title('NRObjects: decoded ' + ' × '.join(names) + ' centers')
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--results_dir', default='results/ENRC_nr_objects')
    parser.add_argument('--output_prefix')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--split', choices=('train', 'test'), default='test')
    parser.add_argument('--experiment', type=int, choices=(1, 2))
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError('--batch_size must be positive')
    model_path = Path(args.model_path).expanduser()
    model, run_args = load_model_from_save_dict(str(model_path), map_location=args.device)
    if run_args.dataset != 'fast_nr_objects':
        raise ValueError('Expected a fast_nr_objects checkpoint')
    names, sizes = checkpoint_factors(run_args, model, args.experiment)
    model = model.to(args.device).eval()
    model.device = torch.device(args.device)
    set_final_progressive_state(model, run_args)
    dataset = Fast_NRObjects(root=getattr(run_args, 'nr_objects_root', 'data/enrc_data/nr_objects'),
                         train=args.split == 'train',
                         train_frac=getattr(run_args, 'nr_objects_train_frac', 0.8),
                         seed=getattr(run_args, 'seed', 42), device='cpu',
                         factors_label_list=names)
    probabilities, labels = infer(model, dataset, run_args.model_type, args.batch_size)
    if labels.ndim != 2 or labels.shape[1] != 2 or not len(labels):
        raise ValueError(f'Expected nonempty [N, 2] labels, got {labels.shape}')
    facets = {name: model.n_clusters_j_list.index(n) for name, n in zip(names, sizes)}
    scores = [[matched_accuracy(probabilities[j].argmax(axis=1), labels[:, k],
                                model.n_clusters_j_list[j], n_classes)
               for k, n_classes in enumerate(sizes)] for j in range(2)]
    chosen = {name: scores[facets[name]][k] for k, name in enumerate(names)}
    report = {
        'checkpoint': str(model_path.resolve()), 'split': args.split,
        'n_samples': len(labels),
        'metric': 'one-to-one Hungarian cluster accuracy, matched on evaluation split',
        'label_names': {name: Fast_NRObjects.LABEL_NAMES[Fast_NRObjects.FACTOR_NAMES.index(name)] for name in names},
        'facet_by_label': facets,
        'by_label': chosen,
        'all_facet_label_accuracies': {
            f'facet{j}': {name: scores[j][k]['accuracy'] for k, name in enumerate(names)}
            for j in range(2)},
        'latent_cluster_centers': {
            f'facet{j}': model.mu_p_z_j_c_j_list[j].detach().cpu().T.tolist()
            for j in range(2)},
    }
    output_dir = Path(args.results_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = model_path.parent.parent.name if model_path.parent.name == 'files' else model_path.parent.name
    prefix = args.output_prefix or run_name
    if not prefix or '/' in prefix or '\\' in prefix:
        raise ValueError('Invalid output prefix')
    report_path = output_dir / f'{prefix}_cluster_accuracy_and_centers.json'
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    atlas_path = output_dir / f'{prefix}_cluster_center_atlas.png'
    draw_atlas(model, names, facets, chosen, atlas_path)
    for name, facet in facets.items():
        result = chosen[name]
        print(f'{name}: facet {facet}, accuracy={result["accuracy"]:.4f} '
              f'({result["correct"]}/{len(labels)}), mapping={result["cluster_to_label"]}')
    print(f'Report: {report_path}\nAtlas: {atlas_path}')

if __name__ == '__main__':
    main()
