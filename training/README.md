# Training Folder

The `training` folder contains scripts and configuration files related to training a neural network model for a specific task. Within this folder, you will find three important files: `augmentation.py`, `train_with_cyclical_augmentations_random_sample_nonfix.py`, and `training_config.txt`.

## Augmentation Script - `augmentation.py`

The `augmentation.py` script includes various data augmentation techniques that will be used during the training process to improve model robustness.


## Training Configuration - `cyclical_training_config.txt`


The `cyclical_training_config.txt` file contains all the training parameters and paths required to train the model using `train_with_cyclical_augmentations_random_sample_nonfix.py`.

You can configure the following parameters within this file:

- `modelfile`: Path to the model file where the trained model will be saved.
- `inputfolder`: Path to the input data folder.
- `maskfolder`: Path to the mask/label data folder.
- `load_model`: Path to a pre-trained model (if transfer learning is desired).
- `start_epochs`: The starting epoch for training.
- `epochs`: Total number of training epochs.
- `steps_per_epoch`: Number of steps per training epoch (number of samples being randomly selected).

Ensure that you have provided accurate paths and parameters in the `cyclical_training_config.txt` file before initiating the training process.

Feel free to explore and customize these training scripts and configuration files to meet your specific training needs.

## Training Script - `train_with_cyclical_augmentations_random_sample_nonfix.py`

The `train_with_cyclical_augmentations_random_sample_nonfix.py` script is used to train the neural network model. You can adjust various training parameters within this script's code, including:

- `batch_size`: Batch size for training.
- `learning_rate`: Learning rate for optimization.
- `cyclical_augmentations`: Number of cyclical augmentations.
- `train_ratio`: Training data ratio (e.g., 0.8 for an 80% training and 20% validation split).
- `num_augmentations`: Number of augmentations to apply in each data augmentation cycles.
- `random_block_sampling_pixel`: Number of pixels to sample randomly.
- `weights`: A list of weights for each input label sample (None if not weighted).

You can modify these parameters to customize the training process according to your specific requirements.

To execute the script, open a terminal or command prompt and run the following command:
    ```bash
    python train_with_cyclical_augmentations_random_sample_nonfix.py

## Training Configuration - `training_config.txt`


The `training_config.txt` file contains all the training parameters and paths required to train the model using `train_with_dynamic_cyclical_augmentations_random_sample_nonfix.py`.

You can configure the following parameters within this file:

- `modelfile`: Path to the model file where the trained model will be saved.
- `inputfolder`: Path to the input data folder.
- `maskfolder`: Path to the mask/label data folder.
- `load_model`: Path to a pre-trained model (if transfer learning is desired).
- `start_epochs`: The starting epoch for training.
- `epochs`: Total number of training epochs.
- `steps_per_epoch`: Number of steps per training epoch (number of samples being randomly selected).

- `modelfile`: Path to the model file where the trained model will be saved.
Example:
modelfile = "/mnt/research-data/chongyuh/3dunet/model/dynamic_cyclical_data_augmentation_new_standardized_stop_janelia/"

- `inputfolder`: Path to the input data folder.
Example:
inputfolder = "/mnt/research-data/chongyuh/3dunet/data/janelia-cosem-datasets/raw/"

- `maskfolder`: `Path to the mask/label data folder.
Example:
maskfolder = "/mnt/research-data/chongyuh/3dunet/data/janelia-cosem-datasets/label_f/"

- `val_inputfolder`: Path to the validation input data folder (optional).
Example:
val_inputfolder = "/mnt/research-data/chongyuh/3dunet/data/janelia-cosem-datasets/val/raw/"

- `val_maskfolder`: Path to the validation mask/label data folder (optional).
Example:
val_maskfolder = "/mnt/research-data/chongyuh/3dunet/data/janelia-cosem-datasets/val/label_f/"

- `load_model`: Path to a pre-trained model (if transfer learning is desired).
Example:
load_model = "None"

- `steps_per_epoch`: Number of steps per training epoch (number of samples being randomly selected).
Example:
steps_per_epoch = 500

Ensure that you have provided accurate paths and parameters in the `training_config.txt` file before initiating the training process.

Feel free to explore and customize these training scripts and configuration files to meet your specific training needs.

## Training Script - `train_with_dynamic_cyclical_augmentations_random_sample_nonfix.py`

The `train_with_dynamic_cyclical_augmentations_random_sample_nonfix.py` script is used to train the neural network model with dynamic cyclical augmentations. You can adjust various training parameters within this script's code, including:

- `batch_size`: Batch size for training.
- `learning_rate`: Learning rate for optimization.
- `cyclical_augmentations`: Number of cyclical augmentations.
- `train_ratio`: Training data ratio (e.g., 0.8 for an 80% training and 20% validation split).
- `num_augmentations`: Number of augmentations to apply in each data augmentation cycles.
- `early_stop_number`: The number of data augmentation cycles without improvement and the number of recovery attempts before stopping.
- `augmentation_threshold_f`: The initial threshold to determine whether to apply data augmentation.(f1)
- `augmentation_threshold_iou`: The initial threshold to determine whether to apply data augmentation.(iou)
- `random_block_sampling_pixel`: Number of pixels to sample randomly.
- `weights`: A list of weights for each input label sample (None if not weighted).

You can modify these parameters to customize the training process according to your specific requirements.

To execute the script, open a terminal or command prompt and run the following command:
    ```bash
    python train_with_dynamic_cyclical_augmentations_random_sample_nonfix.py
