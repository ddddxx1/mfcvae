#!/bin/bash
#SBATCH --job-name=ENRCcmnist-training
#SBATCH --array=0-2
#SBATCH --account=seidl_stud
#SBATCH --partition=minor
#SBATCH --qos=minor_student_prio
#SBATCH --gres=gpu:1
#SBATCH --time=10-00:00:00
#SBATCH --output=../logs/slurm-ENRCcmnist_%A_%a.out
#SBATCH --error=../logs/slurm-ENRCcmnist_%A_%a.err

set -euo pipefail

source /nfs/data8/dong/mfcvae/notify/notify.sh
send_bark "集群任务开始 🔛" \
    "任务: ${SLURM_JOB_NAME}，任务 ID: ${SLURM_JOB_ID}，子任务: ${SLURM_ARRAY_TASK_ID} 已开始。" || echo "开始通知发送失败" >&2


cd /nfs/data8/dong/mfcvae
mkdir -p logs

SEEDS=(0 1 2)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
# Override when submitting, for example:
# OUTPUT_DIR=/nfs/data8/dong/mfcvae/outputs/cmnist_56gray sbatch shell_scripts/run_cmnist_3seed.sh
#OUTPUT_DIR="/nfs/data8/dong/mfcvae/outputs/cmnist_56gray"

echo "Training C-MNIST with seed ${SEED}."

./mfcvae-repo-cluster/bin/python -u train.py \
    --config_args_path configs/enrc_cmnist.yml \
    --seed "${SEED}"
#    --output_dir "${OUTPUT_DIR}"

LOG_FILE="slurm-ENRCcmnist_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
send_bark "集群任务完成 🎉" "任务: ${SLURM_JOB_NAME}，任务 ID: ${SLURM_JOB_ID}，子任务: ${SLURM_ARRAY_TASK_ID}，Seed: ${SEED} 已成功跑完！日志: ${LOG_FILE}" || echo "完成通知发送失败" >&2
