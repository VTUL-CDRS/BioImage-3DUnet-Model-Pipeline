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
  for cc in 300
  do
    echo $thr $cc
    # echo $thr $cc
    # python clear.py --pred-file $RPRED/rheb1/pred.tif --mask-file $RMASK/RhebNeuron1_mask.tif \
    #   --outdir $RPRED/rheb1/ --threshold $thr --clear-size $cc --n-jobs 12
    # 
    # echo $thr $cc
    # python clear.py --pred-file $RPRED/rheb2/pred.tif --mask-file $RMASK/RhebNeuron2_mask.tif \
    #   --outdir $RPRED/rheb2/ --threshold $thr --clear-size $cc --n-jobs 12
    
    python clear.py --pred-file $RPRED/rheb3/pred.tif --mask-file $RMASK/RhebNeuron3_mask.tif \
      --outdir $RPRED/rheb3/ --threshold $thr --clear-size $cc --n-jobs 12
    
    python clear.py --pred-file $RPRED/rheb4/pred.tif --mask-file $RMASK/RhebNeuron4_mask.tif \
      --outdir $RPRED/rheb4/ --threshold $thr --clear-size $cc --n-jobs 12
  done
done
