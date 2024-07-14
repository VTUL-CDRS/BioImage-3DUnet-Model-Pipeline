#!/bin/bash

#SBATCH --account=stgnn
#SBATCH --partition=intel_q
#SBATCH --nodes=1 
#SBATCH --mem=300G
#SBATCH --time=0-02:00:00
#SBATCH --gres=gpu:0

hostname
source ~/.bashrc


conda activate imls

export RMASK=/home/linhan/yinlin/bio/raw_images/control
export RPRED=/home/linhan/yinlin/bio/predictions/mixture

for thr in 30 60 127
do 
  for cc in 300 500
  do
    echo $thr $cc
    python analysis.py --pred-file $RPRED/control2/pred.tif --mask-file $RMASK/ControlNeuron2_mask.tif \
      --outdir $RPRED/control2/ --threshold $thr --clear-size $cc --n-jobs 32
    
    echo $thr $cc
    python analysis.py --pred-file $RPRED/control3/pred.tif --mask-file $RMASK/ControlNeuron3_mask.tif \
      --outdir $RPRED/control3/ --threshold $thr --clear-size $cc --n-jobs 32
  done
done
