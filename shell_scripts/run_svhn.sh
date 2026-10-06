#!/usr/bin/env bash
#
#SBATCH -A seidl_stud
#SBATCH -p minor
#SBATCH --qos=minor_student_prio
#SBATCH --job-name=eval_mnist
#SBATCH --output=res.txt
#SBATCH --ntasks=1
#SBATCH --time=150:00
#SBATCH --gres=gpu:1

cd ..
python3 train.py --config_args_path "configs/svhn.yml"
cd shell_scripts