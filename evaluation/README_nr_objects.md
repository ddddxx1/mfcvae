# ENRC NRObjects: two experiments

Run commands from the repository root. The dataset contains 10,000 single-object
images in `data/enrc_data/nr_objects/images/train`, with matching individual JSON
files in `scenes`. The loader uses individual scenes because the supplied root
aggregate JSON is incomplete. Images are resized from 128×128 to 32×32 RGB.

Like the two 3dshapes configurations, these are independent two-facet experiments:

| Experiment | Config | Labels in order | Clusters |
| --- | --- | --- | --- |
| 1 | `configs/enrc_nr_objects_1.yml` | color, material | 6, 2 |
| 2 | `configs/enrc_nr_objects_2.yml` | color, shape | 6, 3 |

Both use fixed uniform priors (`fix_pi_p_c: true`, `facet_to_fix_pi_p_c: all`),
16 latent dimensions per facet, and two progressive stages of 101 epochs
(202 total). Both retain all images, including variation in the unselected
attribute. Label columns follow `factors_label_list`; integer IDs within each
attribute are defined by `Fast_NRObjects.COLOR_NAMES`, `MATERIAL_NAMES`, and
`SHAPE_NAMES` in `datasets_enrc.py`:

- color: blue, gray, green, purple, red, yellow (IDs 0–5)
- material: metal, rubber (IDs 0–1)
- shape: cube, cylinder, sphere (IDs 0–2)

The YAML `factors_variation_dict` entries now contain attribute names rather
than numeric IDs. Their order does not change these IDs: the NRObjects loader
uses its constants above and does not use `factors_variation_dict` to filter
images or encode labels.

Splitting remains stratified on all three source attributes before selecting
label columns: 7,983 train and 2,017 test images. Both experiments use identical
splits for a given seed. The cache contains all three labels and can be reused
by both experiments. It is stored at
`data/enrc_data/nr_objects/nr_objects_cache_rgb32_v1.pt` (under `nr_objects_root`
if customized). After modifying source data, delete this cache or instantiate
`Fast_NRObjects(force_reprocess=True)` to rebuild it.

## Train

```bash
python train.py --config_args_path configs/enrc_nr_objects_1.yml --seed 0
python train.py --config_args_path configs/enrc_nr_objects_2.yml --seed 0
```

Three-seed Slurm arrays (0, 1, 2) for each experiment:

```bash
sbatch shell_scripts/run_ENRCnr_objects_1_3seed.sh
sbatch shell_scripts/run_ENRCnr_objects_2_3seed.sh
```

The scripts use `/nfs/data8/dong/mfcvae`, its
`mfcvae-repo-cluster/bin/python` environment, and `notify/notify.sh`. Both request
one GPU and a five-hour time limit. Training stdout/stderr paths are
`../logs/slurm-ENRCnr_objects_<experiment>_%A_%a.out` / `.err`, relative to the
submission directory; create that log directory before submitting.

For local CPU execution, copy the YAML and set `device: cpu`, optionally with
`wandb_mode: disabled`. YAML overrides most CLI flags; an explicitly supplied
`--seed` takes precedence over the YAML and W&B sweep seed.

With the supplied online W&B configuration, checkpoints are saved to
`wandb.run.dir/save_dict.pt` when `save_model: true`. At normal completion,
`args.pickle` is written to the same directory even if `save_model: false`.
With W&B disabled, the output directory is `outputs/disabled_seed<seed>/`;
both experiments use the same directory for a given seed, so later runs can
overwrite earlier outputs there.

## Evaluate

```bash
export PYTHONPATH="$PWD:$PWD/evaluation${PYTHONPATH:+:$PYTHONPATH}"
python evaluation/evaluate_nr_objects_cluster_accuracy.py \
  --model_path /path/to/experiment1/save_dict.pt --experiment 1 \
  --results_dir results/ENRC/nr_objects/config1 --output_prefix seed0 --device cpu
python evaluation/evaluate_nr_objects_cluster_accuracy.py \
  --model_path /path/to/experiment2/save_dict.pt --experiment 2 \
  --results_dir results/ENRC/nr_objects/config2 --output_prefix seed0 --device cpu
```

The selected attributes are read from checkpoint args; `--experiment` also
checks that the supplied checkpoint belongs to the intended experiment.
JSON results contain the two selected attribute accuracies, a 2×2 facet–label
accuracy matrix, cluster mappings, contingency tables, sizes, purities and
latent centers. Center atlases have six color rows and either two material
columns or three shape columns. Facets are identified by unique cluster counts;
the full accuracy matrix helps check the learned semantic associations.
Hungarian cluster-to-label mappings are fitted on the evaluation split, as in
the existing GTSRB evaluator. These are clustering scores.

Three-seed evaluation scripts currently contain fixed checkpoint paths for the
November 5 runs, grouped under
`wandb/run-ENRCnrobjects-config1-main-v1/` and
`wandb/run-ENRCnrobjects-config2-main-v1/`. For new runs, edit `SEED0_PATH`,
`SEED1_PATH`, and `SEED2_PATH` **inside the corresponding script**. These are
unconditional assignments: passing same-named environment variables to `sbatch`
does not override them.

From the repository root, create the Slurm log directories before submitting:

```bash
mkdir -p ../logs/cluster_accuracy/nr_objects/config1 \
         ../logs/cluster_accuracy/nr_objects/config2
sbatch shell_scripts/evaluate_nr_objects_cluster_accuracy_1_3seed.sh
sbatch shell_scripts/evaluate_nr_objects_cluster_accuracy_2_3seed.sh
```

The scripts use CUDA, batch size 500, and the test split, and check the experiment
number against checkpoint args. Their default result directories are
`results/ENRC/nr_objects/config1` and `results/ENRC/nr_objects/config2`;
`RESULTS_DIR` can be overridden when submitting. Each seed writes:

- `seed<seed>_cluster_accuracy_and_centers.json`
- `seed<seed>_cluster_center_atlas.png`

Slurm stdout/stderr are written under
`../logs/cluster_accuracy/nr_objects/config<experiment>/`, relative to the
submission directory. The standalone Python evaluator defaults to
`results/ENRC_nr_objects` if `--results_dir` is omitted. Without `--output_prefix`,
it derives the prefix from the checkpoint's run directory.

Reconstruction visualization supports either checkpoint:

```bash
python evaluation/visualize_nr_objects_reconstructions.py \
  --model_path /path/to/save_dict.pt --device cpu \
  --output results/ENRC/nr_objects/config1/seed0_reconstructions.png
```

The reconstruction script defaults to the first 12 examples. Use `--n_examples`
to change the count, or `--indices` to select explicit zero-based indices within
the chosen split. Its default output is
`results/ENRC_nr_objects/reconstructions.png` if `--output` is omitted.

Both evaluation scripts read the dataset root, split fraction, and seed from the
checkpoint configuration and accept `--split train|test` (default test). Neither
currently accepts `--data_root`; make the configured `nr_objects_root` available
when moving checkpoints between machines. Relative dataset paths are resolved
from the working directory. Three-facet checkpoints from the earlier
configuration must be retrained using the new pair configs.
