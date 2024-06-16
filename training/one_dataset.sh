#!/bin/bash

#SBATCH --account=stgnn
#SBATCH --partition=dgx_normal_q
#SBATCH --nodes=1 
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1

hostname
source ~/.bashrc

module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

conda activate imls

python train_with_dynamic_cyclical_data_augmentation_random_sample_nonfix.py --config cfg.txt
