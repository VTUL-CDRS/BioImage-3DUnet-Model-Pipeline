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
    root_dir=../prediction/control2 
    python analysis.py --pred-file $root_dir/predictions/combine.tif --mask-file ~/yinlin/bio/raw_images/control/ControlNeuron2_mask.tif \
      --outdir $root_dir --threshold $thr --clear-size $cc --n-jobs 32
    
    root_dir=../prediction/control3 
    python analysis.py --pred-file $root_dir/predictions/combine.tif --mask-file ~/yinlin/bio/raw_images/control/ControlNeuron3_mask.tif \
      --outdir $root_dir --threshold $thr --clear-size $cc --n-jobs 32
  done
done
