#!/bin/bash
#SBATCH --account=seidl_stud
#SBATCH --partition=minor
#SBATCH --qos=minor_student_prio
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --job-name=supervised_class
#SBATCH --output=../logs/supervised_classification-cmnist56-AddDigit-sample-%j.out
#SBATCH --error=../logs/supervised_classification-cmnist56-AddDigit-sample-%j.err

source /nfs/data8/dong/mfcvae/notify/notify.sh
send_bark "集群任务开始" \
    "任务: ${SLURM_JOB_NAME} 已开始。" || echo "开始通知发送失败" >&2

cd ..
./mfcvae-repo-cluster/bin/python -u supervised_classification.py \
  --model_path \
    wandb/run-20261027_152912-i27oz50s/files/save_dict.pt \
    wandb/run-20261027_152913-lmxniip8/files/save_dict.pt \
    wandb/run-20261027_153402-xpgmwqmv/files/save_dict.pt \
  --device cuda:0 \
  --embedding_mode sample \
  --n_runs 3 \
  --classifier_seed 0

send_bark "集群任务完成 🎉" "任务: ${SLURM_JOB_NAME} 已成功跑完！" || echo "完成通知发送失败" >&2