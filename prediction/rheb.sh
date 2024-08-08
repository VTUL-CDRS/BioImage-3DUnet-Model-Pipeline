#!/bin/bash

#SBATCH --account=chongyu_he
#SBATCH --partition=dgx_normal_q
#SBATCH --nodes=1 
#SBATCH --mem=128G
#SBATCH --time=0-05:00:00
#SBATCH --gres=gpu:1

hostname
source ~/.bashrc

conda activate dgx

export PYTHONPATH=/home/linhan/yinlin/projects/BioImage-3DUnet-Model-Pipeline

export ROOT=/home/linhan/yinlin/bio
export RAW=$ROOT/raw_images/rheb
export PRED=$ROOT/predictions/mixture2

for i in 1 2 3 4 
# for i in 4 
do 
  mkdir $PRED/rheb${i} 
  python predict.py --ckpt ../checkpoints_mixture2/epoch=999-val_dice_loss=0.1094.ckpt --inputfile $RAW/RhebNeuron${i}_Raw.tif --outputfile $PRED/rheb${i}/pred.tif
done

