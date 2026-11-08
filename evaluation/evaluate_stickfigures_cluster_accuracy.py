"""Evaluate ENRC Stickfigures upper/lower posture clustering and decode grayscale cluster centers.

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

from datasets_enrc import Fast_Stickfigures
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


@torch.no_grad()
def decode_atlas(model, upper_facet, lower_facet, upper_order, lower_order):
    centers = [p.detach() for p in model.mu_p_z_j_c_j_list]
    tiles = []
    for upper_cluster in upper_order:
        row = []
        for lower_cluster in lower_order:
            ids = [None, None]
            ids[upper_facet], ids[lower_facet] = upper_cluster, lower_cluster
            z = [centers[j][:, ids[j]].unsqueeze(0) for j in range(2)]
            gray = model.decode(z).reshape(20, 20).clamp(0, 1).cpu().numpy()
            row.append(gray)
        tiles.append(row)
    return tiles


def draw_atlas(tiles, upper_order, lower_order, chosen, output_path):
    """Draw the decoded combinations as one table with row and column labels."""
    mosaic = np.concatenate([np.concatenate(row, axis=1) for row in tiles], axis=0)
    tile_height, tile_width = tiles[0][0].shape[:2]
    n_upper, n_lower = len(upper_order), len(lower_order)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    fig.subplots_adjust(left=0.29, right=0.94, bottom=0.10, top=0.84)
    ax.imshow(mosaic, cmap='gray', vmin=0, vmax=1, interpolation='nearest')

    ax.set_xticks([(col + 0.5) * tile_width - 0.5 for col in range(n_lower)])
    ax.set_xticklabels([
        f'{chosen["lower_body"]["cluster_to_label"][cluster]}\n(c{cluster})'
        for cluster in lower_order
    ], fontsize=11)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Lower body: matched posture (cluster ID)', labelpad=10)

    ax.set_yticks([(row + 0.5) * tile_height - 0.5 for row in range(n_upper)])
    ax.set_yticklabels([
        f'{chosen["upper_body"]["cluster_to_label"][cluster]}\n(c{cluster})'
        for cluster in upper_order
    ], fontsize=10)
    ax.set_ylabel('Upper body: matched posture (cluster ID)', labelpad=10)
    ax.tick_params(axis='both', length=0, pad=9)

    for col in range(1, n_lower):
        ax.axvline(col * tile_width - 0.5, color='white', linestyle=':', linewidth=1)
    for row in range(1, n_upper):
        ax.axhline(row * tile_height - 0.5, color='white', linestyle=':', linewidth=1)

    fig.suptitle('Stickfigures: decoded upper × lower posture cluster centers', fontsize=16, y=0.97)
    fig.text(0.5, 0.035,
             'Each cell = decoder(upper center, lower center). Labels come from evaluation matching.',
             ha='center', fontsize=9)
    fig.savefig(output_path, dpi=180, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--results_dir', default='results/ENRC_stickfigures')
    parser.add_argument('--output_prefix')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--split', choices=('train', 'test'), default='test')
    parser.add_argument('--data_file', help='Override the checkpoint dataset file for loading and inference')
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError('--batch_size must be positive')
    model_path = Path(args.model_path).expanduser()
    model, run_args = load_model_from_save_dict(
        str(model_path), map_location=args.device, stickfigures_file=args.data_file)
    if run_args.dataset != 'fast_stickfigures' or model.J_n_mixtures != 2:
        raise ValueError('Expected a two-facet fast_stickfigures checkpoint')
    if list(model.n_clusters_j_list) != [3, 3]:
        raise ValueError('Expected 3 upper-body clusters and 3 lower-body clusters')
    model = model.to(args.device).eval()
    model.device = torch.device(args.device)
    set_final_progressive_state(model, run_args)
    data_file = Path(getattr(run_args, 'stickfigures_file',
                            'data/enrc_data/stickfigures/stickfigures_3sub.data')).expanduser()
    dataset = Fast_Stickfigures(
        data_file=data_file, train=args.split == 'train',
        train_frac=getattr(run_args, 'stickfigures_train_frac', 0.8),
        seed=getattr(run_args, 'seed', 42), device='cpu')
    if not len(dataset):
        raise ValueError('Expected a nonempty evaluation split')
    probabilities, labels = infer(model, dataset, run_args.model_type, args.batch_size)
    if labels.ndim != 2 or labels.shape[1] != 2 or not len(labels):
        raise ValueError(f'Expected nonempty [N, 2] labels, got {labels.shape}')
    scores = [[matched_accuracy(probabilities[j].argmax(axis=1), labels[:, k],
                                model.n_clusters_j_list[j], n_classes)
               for k, n_classes in enumerate((3, 3))] for j in range(2)]
    # Both facets have three clusters, so cluster count cannot identify semantics.
    # Assign distinct facets to labels by maximum total matched accuracy.
    accuracy_matrix = np.array([[score['accuracy'] for score in row] for row in scores])
    facet_rows, label_cols = linear_sum_assignment(-accuracy_matrix)
    facet_for_label = dict(zip(label_cols.tolist(), facet_rows.tolist()))
    upper_facet, lower_facet = facet_for_label[0], facet_for_label[1]
    chosen = {'upper_body': scores[upper_facet][0], 'lower_body': scores[lower_facet][1]}
    report = {
        'checkpoint': str(model_path.resolve()), 'split': args.split,
        'n_samples': len(labels),
        'metric': 'one-to-one Hungarian cluster accuracy, matched on evaluation split',
        'label_names': list(Fast_Stickfigures.FACTOR_NAMES),
        'class_ids': {'upper_body': [0, 1, 2], 'lower_body': [0, 1, 2]},
        'data_file': str(data_file.resolve()),
        'seed': getattr(run_args, 'seed', 42),
        'train_frac': getattr(run_args, 'stickfigures_train_frac', 0.8),
        'facet_assignment': 'one-to-one maximum total accuracy on evaluation split',
        'facet_for_upper_body': upper_facet, 'facet_for_lower_body': lower_facet,
        'by_label': chosen,
        'all_facet_label_accuracies': {
            f'facet{j}': {'upper_body': scores[j][0]['accuracy'], 'lower_body': scores[j][1]['accuracy']}
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

    upper_order = sorted(range(3), key=lambda c: chosen['upper_body']['cluster_to_label'][c])
    lower_order = sorted(range(3), key=lambda c: chosen['lower_body']['cluster_to_label'][c])
    tiles = decode_atlas(model, upper_facet, lower_facet, upper_order, lower_order)
    atlas_path = output_dir / f'{prefix}_cluster_center_atlas.png'
    draw_atlas(tiles, upper_order, lower_order, chosen, atlas_path)
    for name, facet in (('upper_body', upper_facet), ('lower_body', lower_facet)):
        result = chosen[name]
        print(f'{name}: facet {facet}, accuracy={result["accuracy"]:.4f} '
              f'({result["correct"]}/{len(labels)}), mapping={result["cluster_to_label"]}')
    print(f'Report: {report_path}\nAtlas: {atlas_path}')

if __name__ == '__main__':
    main()
