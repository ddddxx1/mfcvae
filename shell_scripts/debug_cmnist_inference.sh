#!/bin/bash
#SBATCH --account=seidl_stud
#SBATCH --partition=minor
#SBATCH --qos=minor_student_prio
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --job-name=debug_ENRCcmnist_truefixpipc_seed2_inference
#SBATCH --exclude=worker-minor-6
#SBATCH --output=../logs/debug_ENRCcmnist_truefixpipc_seed2_inference-%j.out
#SBATCH --error=../logs/debug_ENRCcmnist_truefixpipc_seed2_inference-%j.err

source /nfs/data8/dong/mfcvae/notify/notify.sh
send_bark "集群任务开始" \
    "任务: ${SLURM_JOB_NAME} 已开始。"

# 激活你的环境
source /nfs/data8/dong/mfcvae/mfcvae-repo-cluster/bin/activate # 或者 conda activate mfcvae-repo-cluster
cd ..
# 运行代码
python debug_cmnist_inference.py \
  --model_path  \
    wandb/run-ENRCcmnist-truefixpipc-seed2/files/save_dict.pt \
  --device cuda:0

send_bark "集群任务完成" "任务: ${SLURM_JOB_NAME} 已成功跑完！"
