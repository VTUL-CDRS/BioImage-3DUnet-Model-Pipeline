# Test Folder

The `test` folder contains various scripts used for testing different aspects of a trained neural network model. Within this folder, you will find six important files: `augmentation.py`, `train_with_augmentations_test.py`, `train_with_cyclical_augmentations.py`, `train_with_random_sample.py`, `train_with_augmentations_random_sample.py`, and `train_with_cyclical_augmentations_random_sample.py`.

## Augmentation Script - `augmentation.py`

The `augmentation.py` script includes various data augmentation techniques that can be used during testing to evaluate the model's performance.

## Testing Script - `train_with_augmentations_test.py`

The `train_with_augmentations_test.py` script is exclusively used to test each augmentation technique. It could be used when you want to understand the impact of each individual augmentation technique on the performance of your model.

## Testing Script - `train_with_cyclical_augmentations.py`

The `train_with_cyclical_augmentations.py` script is used to test the model's performance when using cyclical augmentations.

## Testing Script - `train_with_random_sample.py`

The `train_with_random_sample.py` script is used to evaluate the impact of random block sampling on the model's performance during training.

## Testing Script - `train_with_augmentations_random_sample.py`

The `train_with_augmentations_random_sample.py` script combines augmentations and random block sampling during testing. This script allows you to assess the model's performance under both augmentation and random sampling conditions.

## Testing Script - `train_with_cyclical_augmentations_random_sample.py`

The `train_with_cyclical_augmentations_random_sample.py` script is used to test the model's performance when both cyclical augmentations and random block sampling are applied. You can evaluate how the combination of these techniques affects model performance.

Each script contains predefined testing parameters, but you can customize these parameters according to your specific testing requirements. After configuring the parameters, you can run each script using the following command:

    ```bash
    python script_name.py

Replace script_name.py with the name of the script you want to run.

Feel free to explore and modify these testing scripts to evaluate your trained model's performance under various conditions.