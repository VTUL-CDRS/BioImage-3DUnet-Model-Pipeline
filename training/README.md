# Training Folder

The `training` folder contains scripts and configuration files related to training a neural network model for a specific task. Within this folder, you will find three important files: `augmentation.py`, `train_with_cyclical_augmentations_random_sample_nonfix.py`, and `training_config.txt`.

## Augmentation Script - `augmentation.py`

The `augmentation.py` script includes various data augmentation techniques that will be used during the training process to improve model robustness.


## Training Configuration - `training_config.txt`

The `training_config.txt` file contains all the training parameters and paths required to train the model. You can configure the following parameters within this file:

- `modelfile`: Path to the model file where the trained model will be saved.
- `inputfolder`: Path to the input data folder.
- `maskfolder`: Path to the mask/label data folder.
- `load_model`: Path to a pre-trained model (or "None" if not desired).
- `start_epochs`: The starting epoch for training.
- `epochs`: Total number of training epochs.
- `steps_per_epoch`: Number of steps per training epoch (number of samples being randomly selected).

Ensure that you have provided accurate paths and parameters in the `training_config.txt` file before initiating the training process.

Feel free to explore and customize these training scripts and configuration files to meet your specific training needs.

## Training Script - `train_with_cyclical_augmentations_random_sample_nonfix.py`

The `train_with_cyclical_augmentations_random_sample_nonfix.py` script is used to train the neural network model. You can adjust various training parameters within this script's code, including:

- `batch_size`: Batch size for training.
- `learning_rate`: Learning rate for optimization.
- `cyclical_augmentations`: Number of cyclical augmentations.
- `train_ratio`: Training data ratio (e.g., 0.8 for an 80% training and 20% validation split).
- `num_augmentations`: Number of augmentations to apply.
- `random_block_sampling_pixel`: Number of pixels to sample randomly.
- `weights`: A list of weights for each input label sample (None if not weighted).

You can modify these parameters to customize the training process according to your specific requirements.

To execute the script, open a terminal or command prompt and run the following command:
    ```bash
    python train_with_cyclical_augmentations_random_sample_nonfix.py

- ## Data and Label

- If the labeled image does not have the same name as the raw TIF image, please name them with a similar number to help the program point to the correct image and label, as mentioned in the appendix of the paper.

|-- data
| |-- Ground_Truth_Method_1
| | |-- 00_RhebStandardCrop2_1_Pore.tif
| | |-- 01_RhebStandardCrop2_2_Pore.tif
| | |-- 02_RhebStandardCrop2_3_Pore.tif
| | |-- 03_RhebStandardCrop2_4_Pore.tif
| | |-- 04_RhebStandardCrop2_5_Pore.tif
| | |-- 05_RhebStandardCrop2_6_Pore.tif
| | |-- 06_RhebStandardCrop2_7_Pore.tif
| | |-- 07_RhebStandardCrop2_8_Pore.tif
| | |-- 08_RhebStandardCrop3_1_Pores.tif
| | |-- 09_RhebStandardCrop3_2_Pores.tif
| | |-- 10_RhebStandardCrop3_3_Pores.tif
| | |-- 11_RhebStandardCrop3_4_Pores.tif
| | |-- 12_RhebStandardCrop4_1_Pores.tif
| | |-- 13_RhebStandardCrop4_2_Pores.tif
| | |-- 14_RhebStandardCrop4_3_Pores.tif
| | -- 15_RhebStandardCrop4_4_Pores.tif | -- raw
| |-- 00_RhebStandardCrop2_1.tif
| |-- 01_RhebStandardCrop2_2.tif
| |-- 02_RhebStandardCrop2_3.tif
| |-- 03_RhebStandardCrop2_4.tif
| |-- 04_RhebStandardCrop2_5.tif
| |-- 05_RhebStandardCrop2_6.tif
| |-- 06_RhebStandardCrop2_7.tif
| |-- 07_RhebStandardCrop2_8.tif
| |-- 08_RhebStandardCrop3_1.tif
| |-- 09_RhebStandardCrop3_2.tif
| |-- 10_RhebStandardCrop3_3.tif
| |-- 11_RhebStandardCrop3_4.tif
| |-- 12_RhebStandardCrop4_1.tif
| |-- 13_RhebStandardCrop4_2.tif
| |-- 14_RhebStandardCrop4_3.tif
| `-- 15_RhebStandardCrop4_4.tif
