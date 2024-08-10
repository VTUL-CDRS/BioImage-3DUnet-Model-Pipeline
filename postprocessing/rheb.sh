#!/bin/bash

#SBATCH --account=chongyu_he
#SBATCH --partition=intel_q
#SBATCH --nodes=1 
#SBATCH --mem=300G
#SBATCH --time=0-05:00:00
#SBATCH --gres=gpu:0

hostname
source ~/.bashrc


conda activate imls

export RMASK=/home/linhan/yinlin/bio/raw_images/rheb
export RPRED=/home/linhan/yinlin/bio/predictions/mixture2

for thr in 127
do 
  for cc in 200 
  do
    for i in 1 2 3 4
    do
      echo $thr $cc $i
      python analysis.py --pred-file $RPRED/rheb${i}/pred.tif --mask-file $RMASK/RhebNeuron${i}_mask.tif \
        --outdir $RPRED/rheb${i}/ --threshold $thr --clear-size $cc --n-jobs 16
    done    
  done
done
