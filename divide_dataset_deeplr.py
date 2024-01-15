import os
import shutil
import random

def copy_and_delete_files(input_dir, output_dir, fraction=0.2):
    """
    Copies a fraction of files from input_dir to output_dir and deletes the original files.

    Parameters:
    input_dir (str): Path to the directory containing the original files.
    output_dir (str): Path to the directory where files will be copied.
    fraction (float): Fraction of files to be copied and deleted (default is 0.2 or 1/5).
    """
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    num_files_to_copy = int(len(files) * fraction)
    files_to_copy = random.sample(files, num_files_to_copy)

    for file in files_to_copy:
        src_path = os.path.join(input_dir, file)
        dst_path = os.path.join(output_dir, file)
        shutil.copy2(src_path, dst_path)
        os.remove(src_path)

copy_and_delete_files(r'dataset/DATASET_DEEPLR/dataset_labeled_divided_nonoveralped_TRAIN_DEEPLR', r'dataset/DATASET_DEEPLR/dataset_labeled_divided_nonoveralped_VAL_DEEPLR')
