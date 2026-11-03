#!/bin/bash
#SBATCH --job-name=ENRCgtsrb-cluster-acc
#SBATCH --array=0-2
#SBATCH --account=seidl_stud
#SBATCH --partition=minor
#SBATCH --qos=minor_student_prio
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --output=../logs/slurm-ENRCgtsrb-cluster-acc-%A_%a.out
#SBATCH --error=../logs/slurm-ENRCgtsrb-cluster-acc-%A_%a.err

set -euo pipefail

source /nfs/data8/dong/mfcvae/notify/notify.sh
send_bark "集群任务开始 🔛" \
    "任务: ${SLURM_JOB_NAME}，任务 ID: ${SLURM_JOB_ID}，子任务: ${SLURM_ARRAY_TASK_ID} 已开始。" \
    || echo "开始通知发送失败" >&2

cd /nfs/data8/dong/mfcvae
export PYTHONPATH="$PWD:$PWD/evaluation${PYTHONPATH:+:$PYTHONPATH}"
RESULTS_DIR="${RESULTS_DIR:-$PWD/results/ENRC/gtsrb}"

# Checkpoints for array tasks 0, 1, and 2. Edit these paths when using new runs.
SEED0_PATH="wandb/run-ENRCgtsrb-main-v1/run-20261103_165056-die055e7/files/save_dict.pt"
SEED1_PATH="wandb/run-ENRCgtsrb-main-v1/run-20261103_165058-ybzkk88o/files/save_dict.pt"
SEED2_PATH="wandb/run-ENRCgtsrb-main-v1/run-20261103_165156-y73kd9sg/files/save_dict.pt"
MODEL_PATHS=("$SEED0_PATH" "$SEED1_PATH" "$SEED2_PATH")
SEED="${SLURM_ARRAY_TASK_ID}"
MODEL_PATH="${MODEL_PATHS[$SEED]}"
if [[ -z "$MODEL_PATH" || ! -f "$MODEL_PATH" ]]; then
    echo "Missing checkpoint for seed ${SEED}: ${MODEL_PATH}. Edit SEED${SEED}_PATH in this script." >&2
    exit 1
fi
mkdir -p "$RESULTS_DIR"
./mfcvae-repo-cluster/bin/python -u evaluation/evaluate_gtsrb_cluster_accuracy.py \
    --model_path "$MODEL_PATH" \
    --results_dir "$RESULTS_DIR" \
    --output_prefix "seed${SEED}" \
    --device cuda:0 \
    --batch_size 500 \
    --split test

LOG_FILE="slurm-ENRCgtsrb-cluster-acc-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
send_bark "集群任务完成 🎉" \
    "任务: ${SLURM_JOB_NAME}，任务 ID: ${SLURM_JOB_ID}，子任务: ${SLURM_ARRAY_TASK_ID}，Seed: ${SEED} 已成功跑完！结果: ${RESULTS_DIR}，日志: ${LOG_FILE}" \
    || echo "完成通知发送失败" >&2
