#!/bin/bash

#SBATCH --account=stgnn
#SBATCH --partition=intel_q
#SBATCH --nodes=1 
#SBATCH --mem=300G
#SBATCH --time=0-05:00:00
#SBATCH --gres=gpu:0

hostname
source ~/.bashrc


conda activate imls

for thr in 30 60 127
do 
  for cc in 300 500
  do
    echo $thr $cc
    root_dir=../prediction/rheb1
    python analysis.py --pred-file $root_dir/predictions/combine.tif --mask-file ~/yinlin/bio/raw_images/rheb/RhebNeuron1_mask.tif \
      --outdir $root_dir --threshold $thr --clear-size $cc --n-jobs 16

    root_dir=../prediction/rheb2 
    python analysis.py --pred-file $root_dir/predictions/combine.tif --mask-file ~/yinlin/bio/raw_images/rheb/RhebNeuron2_mask.tif \
      --outdir $root_dir --threshold $thr --clear-size $cc --n-jobs 16
  done
done
