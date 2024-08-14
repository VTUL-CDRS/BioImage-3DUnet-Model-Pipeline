#!/bin/bash

#SBATCH --account=chongyu_he
#SBATCH --partition=dgx_normal_q
#SBATCH --nodes=1 
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1

hostname
source ~/.bashrc

module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

conda activate cdrs

export PYTHONPATH=../

python train.py --model-dir ./models --data-dir /projects/yinlin_chen/linhan/bio/train/mix_method1
