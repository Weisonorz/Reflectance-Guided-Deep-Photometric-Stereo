#!/bin/bash

# SLURM Resource Parameters

#SBATCH -N 1
#SBATCH --ntasks-per-node=1 # Updated by Yaoyu on Aug. 15, 2023
#SBATCH --cpus-per-task=5
#SBATCH -t 2-00:00 # D-HH:MM
#SBATCH -p a100-gpu-full
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --job-name=PS_FCN_CBN_train
#SBATCH -o job_%j.out
#SBATCH -e job_%j.err

# Executable
# EXE=/bin/bash

source /data2/datasets/ruoguli/miniconda/etc/profile.d/conda.sh
conda activate torch_env
python train_PSFCN.py --in_img_num 32 --dataset PS_Blobby_BRDF_Dataset --model PS_FCN_CBN --retrain /data2/datasets/ruoguli/idl_project_datas/PS-FCN_B_S_32.pth.tar 
