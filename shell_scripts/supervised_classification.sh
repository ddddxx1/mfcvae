#!/bin/bash
#SBATCH --account=seidl_stud
#SBATCH --partition=minor
#SBATCH --qos=minor_student_prio
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --job-name=supervised_class
#SBATCH --output=../logs/supervised_classification/supervised_classification-ENRCstickfigures-%j.out
#SBATCH --error=../logs/supervised_classification/supervised_classification-ENRCstickfigures-%j.err

set -euo pipefail

cd /nfs/data8/dong/mfcvae
PYTHON_BIN="$PWD/mfcvae-repo-cluster/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python interpreter not found: $PYTHON_BIN" >&2
    exit 1
fi
export PYTHONPATH="$PWD:$PWD/evaluation${PYTHONPATH:+:$PYTHONPATH}"

source /nfs/data8/dong/mfcvae/notify/notify.sh
send_bark "集群任务开始" \
    "任务: ${SLURM_JOB_NAME} 已开始。" || echo "开始通知发送失败" >&2

"$PYTHON_BIN" -u evaluation/supervised_classification.py \
  --model_path \
    wandb/run-ENRCstickfigures-main-v1/run-20261107_164418-36pvhgey/files/save_dict.pt \
    wandb/run-ENRCstickfigures-main-v1/run-20261107_164432-ovzlubuh/files/save_dict.pt \
    wandb/run-ENRCstickfigures-main-v1/run-20261107_164432-uex6dai8/files/save_dict.pt \
  --device cuda:0 \
  --embedding_mode sample \
  --n_runs 3 \
  --classifier_seed 0

send_bark "集群任务完成 🎉" "任务: ${SLURM_JOB_NAME} 已成功跑完！" || echo "完成通知发送失败" >&2
