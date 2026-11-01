#!/bin/bash
#SBATCH --job-name=ENRCcmnist-cluster-acc
#SBATCH --array=0-2
#SBATCH --account=seidl_stud
#SBATCH --partition=minor
#SBATCH --qos=minor_student_prio
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --output=../logs/slurm-ENRCcmnist-cluster-acc-%A_%a.out
#SBATCH --error=../logs/slurm-ENRCcmnist-cluster-acc-%A_%a.err

set -euo pipefail

source /nfs/data8/dong/mfcvae/notify/notify.sh
send_bark "集群任务开始 🔛" \
    "任务: ${SLURM_JOB_NAME}，任务 ID: ${SLURM_JOB_ID}，子任务: ${SLURM_ARRAY_TASK_ID} 已开始。" \
    || echo "开始通知发送失败" >&2

cd /nfs/data8/dong/mfcvae

# Fill in the three checkpoint paths before submitting this script.
# Paths are relative to /nfs/data8/dong/mfcvae (or use absolute paths).
SEED0_PATH="wandb/run-ENRCcmnist-main-v1/run-20261030_114449-tu3hk2xr/files/save_dict.pt"
SEED1_PATH="wandb/run-ENRCcmnist-main-v1/run-20261030_114657-7rtentos/files/save_dict.pt"
SEED2_PATH="wandb/run-ENRCcmnist-main-v1/run-20261030_114657-bygy2bh1/files/save_dict.pt"

MODEL_PATHS=("$SEED0_PATH" "$SEED1_PATH" "$SEED2_PATH")
SEED="${SLURM_ARRAY_TASK_ID}"
MODEL_PATH="${MODEL_PATHS[$SEED]}"

if [[ -z "$MODEL_PATH" || ! -f "$MODEL_PATH" ]]; then
    echo "Missing checkpoint for seed ${SEED}: edit SEED${SEED}_PATH in this script to point to an existing save_dict.pt" >&2
    exit 1
fi

# The evaluation script imports modules from both the repository root and
# evaluation/, so both directories must be on Python's import path.
export PYTHONPATH="$PWD:$PWD/evaluation${PYTHONPATH:+:$PYTHONPATH}"
RESULTS_DIR="/nfs/data8/dong/mfcvae/results/ENRC"

echo "Evaluating seed ${SEED}: ${MODEL_PATH}"
echo "Results: ${RESULTS_DIR}"

./mfcvae-repo-cluster/bin/python -u evaluation/evaluate_cmnist_cluster_accuracy.py \
    --model_path "$MODEL_PATH" \
    --results_dir "$RESULTS_DIR" \
    --output_prefix "seed${SEED}" \
    --device cuda:0 \
    --batch_size 500 \
    --split test  # 评测测试集

LOG_FILE="slurm-ENRCcmnist-cluster-acc-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
send_bark "集群任务完成 🎉" \
    "任务: ${SLURM_JOB_NAME}，任务 ID: ${SLURM_JOB_ID}，子任务: ${SLURM_ARRAY_TASK_ID}，Seed: ${SEED} 已成功跑完！结果: ${RESULTS_DIR}，日志: ${LOG_FILE}" \
    || echo "完成通知发送失败" >&2
