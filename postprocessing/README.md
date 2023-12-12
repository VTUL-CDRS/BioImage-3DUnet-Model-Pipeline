# Postprocessing Folder

Within the `postprocessing` folder, you will find two important scripts: `clear.py` and `count.py`. These scripts are used for postprocessing the results obtained after running predictions, specifically for cleaning inaccurate or small predictions and counting the number of nuclear pores in each small predicted result.

## Clear Script - `clear.py`

The `clear.py` script is designed for batch cleaning of predicted results, removing inaccurate or small predictions.

### Usage:

To use the `clear.py` script, follow these steps:

1. Set the input paths, file prefix name, number of files, and small object size that you want to remove in the script:

   ```python
    input_path = r"/mnt/research-data/data/Health-Cell-cut/Health-Cell-full-1/"
    file_name = "Health-Cell-full-1-"
    number = 22
    clear_size = 400

2. To execute the script, open a terminal or command prompt and run the following command:
    ```bash
    python clear.py

## Count Script - `count.py`

The `count.py` script is used to count the number of nuclear pores in individual small predicted result files.

### Usage:

To use the `count.py` script, follow these steps:

1. Set the input paths, file prefix name, number of files in the script:

   ```python
    input_path = r"/mnt/research-data/data/Health-Cell-cut/Health-Cell-full-1/"
    file_name = "clear_400_Health-Cell-full-1-"
    number = 22

2. To execute the script, open a terminal or command prompt and run the following command:
    ```bash
    python count.py

The result of the counting will be printed out in the terminal or command prompt.