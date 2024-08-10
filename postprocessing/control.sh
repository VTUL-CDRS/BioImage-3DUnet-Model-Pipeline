#!/bin/bash

#SBATCH --account=chongyu_he
#SBATCH --partition=intel_q
#SBATCH --nodes=1 
#SBATCH --mem=300G
#SBATCH --time=0-02:00:00
#SBATCH --gres=gpu:0

hostname
source ~/.bashrc


conda activate imls

export RMASK=/home/linhan/yinlin/bio/raw_images/control
export RPRED=/home/linhan/yinlin/bio/predictions/mixture2

for thr in 127
do 
  for cc in 200 
  do
    for i in 1 2 3 4 8
    do
      echo $thr $cc $i
      python analysis.py --pred-file $RPRED/control${i}/pred.tif --mask-file $RMASK/ControlNeuron${i}_mask.tif \
        --outdir $RPRED/control${i}/ --threshold $thr --clear-size $cc --n-jobs 32
    done
  done
done
