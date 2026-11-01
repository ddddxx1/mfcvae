#!/bin/bash
#SBATCH --job-name=cmnist_recon
#SBATCH --account=seidl_stud
#SBATCH --partition=minor
#SBATCH --qos=minor_student_prio
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --exclude=worker-minor-6
#SBATCH --output=../logs/cmnist_recon-%j.out
#SBATCH --error=../logs/cmnist_recon-%j.err

cd ..
python visualize_cmnist_reconstructions.py \
  --model_path outputs/disabled_seed0/save_dict.pt \
  --device cuda:0 \
  --output results/cmnist_reconstructions_seed0_50.png \
  --split test \
  --n_examples 50