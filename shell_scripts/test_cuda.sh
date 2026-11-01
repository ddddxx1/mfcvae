#!/bin/bash
#SBATCH --job-name=test_cuda
#SBATCH --account=seidl_stud
#SBATCH --partition=minor
#SBATCH --qos=minor_student_prio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=../logs/test_cuda-%j.out
#SBATCH --error=../logs/test_cuda-%j.err

source /nfs/data8/dong/mfcvae/notify/notify.sh
send_bark "集群任务开始" \
    "任务: ${SLURM_JOB_NAME} 已开始。"

cd /nfs/data8/dong/mfcvae

echo "========== NODE =========="
hostname

echo "========== CUDA ENV =========="
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo "========== NVIDIA SMI =========="
nvidia-smi

echo "========== PYTORCH =========="
./mfcvae-repo-cluster/bin/python -c "
import torch

print('PyTorch version:', torch.__version__)
print('CUDA version:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())

if torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(0))
    x = torch.randn(100, 100, device='cuda:0')
    print('GPU computation successful:', (x @ x).shape)
"

squeue -h -r -u "$(id -un)" -t R,PD -o '%t' >&2 \
    || echo "squeue 退出码: $?" >&2

send_bark "集群任务完成" "任务: ${SLURM_JOB_NAME} 已成功跑完！"
