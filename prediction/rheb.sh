#!/bin/bash

#SBATCH --account=stgnn
#SBATCH --partition=dgx_normal_q
#SBATCH --nodes=1 
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1

hostname
source ~/.bashrc

module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

conda activate imls

mkdir rheb1 
python predict.py ../training/exp_rheb_method1/rheb_best.h5 ~/yinlin/bio/raw_images/rheb/RhebNeuron1_Raw.tif rheb1/

mkdir rheb2 
python predict.py ../training/exp_rheb_method1/rheb_best.h5 ~/yinlin/bio/raw_images/rheb/RhebNeuron2_Raw.tif rheb2/

mkdir rheb3
python predict.py ../training/exp_rheb_method1/rheb_best.h5 ~/yinlin/bio/raw_images/rheb/RhebNeuron3_Raw.tif rheb3/

mkdir rheb4
python predict.py ../training/exp_rheb_method1/rheb_best.h5 ~/yinlin/bio/raw_images/rheb/RhebNeuron4_raw.tif rheb4/
