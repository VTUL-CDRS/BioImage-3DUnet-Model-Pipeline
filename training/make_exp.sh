#!/bin/bash

export EXP=exp_$@
mkdir $EXP
cp train_with_dynamic_cyclical_data_augmentation_random_sample_nonfix.py $EXP/
cp augmentation.py $EXP/
cp cfg.txt $EXP/
cp one_dataset.sh $EXP/
