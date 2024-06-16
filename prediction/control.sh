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

mkdir control1
python predict.py ../training/exp_control_method1/control_best.h5 ~/yinlin/bio/raw_images/control/ControlNeuron1_Raw-001.tif control1/

mkdir control2
python predict.py ../training/exp_control_method1/control_best.h5 ~/yinlin/bio/raw_images/control/ControlNeuron2_Raw-002.tif control2/

mkdir control3
python predict.py ../training/exp_control_method1/control_best.h5 ~/yinlin/bio/raw_images/control/ControlNeuron3_Raw-004.tif control3/

mkdir control4
python predict.py ../training/exp_control_method1/control_best.h5 ~/yinlin/bio/raw_images/control/ControlNeuron4_Raw-003.tif control4/
