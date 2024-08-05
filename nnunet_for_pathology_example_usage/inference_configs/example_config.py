import os
from pathlib import Path
import pandas as pd
import torch # for norm

#################################################################
### Functions and utilities
#################################################################
current_os = "w" if os.name == "nt" else "l"
other_os = "l" if current_os == "w" else "w"

def convert_path(path, to=current_os):
    '''
    This function converts paths to the format of the desired platform. 
    By default it changes paths to the platform you are currently using.
    This way you do not need to change your paths if you are executing your code on a different platform.
    It may however be that you mounted the drives differently. 
    In that case you may need to change that in the code below. 
    For example, change the Z for blissey (where I mounted it) to X (how you may have mounted it)
    If you add a linux path on a linux platform, or windows on windows, this function by default doesnt do anything.
    '''
    if to in ["w", "win", "windows"]:
        path = path.replace("/data/pathology", "Z:")
        path = path.replace("/data/pa_cpgarchive1", "W:")
        path = path.replace("/data/pa_cpgarchive2", "X:")
        path = path.replace("/data/pa_cpg", "Y:")
        path = path.replace("/data/temporary", "T:")
        path = path.replace("/", "\\")
    if to in ["u", "unix", "l", "linux"]:
        path = path.replace("Z:", "/data/pathology")
        path = path.replace("W:", "/data/pa_cpgarchive1")
        path = path.replace("X:", "/data/pa_cpgarchive2")
        path = path.replace("Y:", "/data/pa_cpg")
        path = path.replace("T:", "/data/temporary")
        path = path.replace("\\", "/")
    return path

def norm_01(x_batch): # Use this for models trained on 0-1 scaled data
    x_batch = x_batch / 255
    x_batch = x_batch.transpose(3, 0, 1, 2)
    return x_batch

def z_norm(x_batch): # use this for default nnunet models, using z-score normalized data
    mean = x_batch.mean(axis=(-2,-1), keepdims=True)
    std = x_batch.std(axis=(-2,-1), keepdims=True)
    x_batch = ((x_batch - mean) / (std + 1e-8))
    x_batch = x_batch.transpose(3, 0, 1, 2)
    return x_batch

def csv_to_matches(path_df):
    """
    This function reads a csv file containing image and mask paths and returns a list of tuples.
    Each tuple contains the image path and the mask path.
    """
    df = pd.read_csv(path_df)
    df = df.drop_duplicates()
    if len(df.columns) > 2:
        raise ValueError("The DataFrame should have only 2 columns: first contains image path and second contains mask path.")
    matches = list(df.itertuples(index=False, name=None))
    return matches

def return_matches_to_run(matches, output_folder):
    """
    The whole pipeline is costructed in such a way that a <name>_runtime.txt file is created once its
    completely done for this WSI + mask match. This means that this file should not be 
    processed anymore by another python script that is run in parralel (also doesnt need to be copied 
    to the local machine (i.e. our computing cluster)). If rerun_unfinished is set to False, the 
    loop will still check if the WSI + mask match is already initiaded, meaning that they are probably
    being processed by another python script. In this case still not 'matches_to_run' will be run. 
    If rerun_unfinished is set to True, the loop will not check if the WSI + mask match is already 
    initiaded and will process and run all 'matches_to_run'.
    """
    runtime_stems = [file[:-12] for file in os.listdir(output_folder) if file.endswith('_runtime.txt')]
    imgs, _ = zip(*matches)
    img_stems = [Path(file).stem for file in imgs]
    matches_to_run_idx = [i for i in range(len(img_stems)) if img_stems[i] not in runtime_stems]
    matches_to_run = [matches[i] for i in matches_to_run_idx]

    if len(matches_to_run) == 0:
        print(f"\nAll files have been processed already, see {output_folder}")
    else:
        print(f"\nReturning {len(matches_to_run)} matches that are not finished yet")
    return matches_to_run

#################################################################
### SET CONFIG
#################################################################

### TASK AND MODEL STUFF'
model_base_path = convert_path('/data/temporary/joey/recovery/nnUNet_raw_data_base/results/nnUNet/2d/Task032_batch1_2_3_4_6_to_train_roi_based_split_NEW/nnUNetTrainerV2_BN_pathology_DA_ignore0_hed005__nnUNet_RGB_scaleTo_0_1_bs8_ps512')

norm = norm_01 # z_norm or norm_01, norm_01 for models trained on 0-1 scaled data, z_norm for default nnunet models, using z-score normalized data 
output_minus_1 = True # Set to True if you want to subtract 1 from the argmax (for example when label 0 is ignore and label 1 is background, which you want to be 0 for visualisation purposes)

### OUTPUT FOLDER AND DATASET NAME
output_folder = Path(f'/data/temporary/joey/data/nnunet_v1_test')
local_output_folder = Path('/home/user/workdir')
os.makedirs(output_folder, exist_ok=True)
os.makedirs(local_output_folder, exist_ok=True)

### MATCHES YOU WANT TO RUN
# This is a list of tuples, where each tuple contains the path to the wsi and the path to the tissue mask
# Example 1: [('path/to/wsi1', 'path/to/tissue_mask1'), ('path/to/wsi2', 'path/to/tissue_mask2')]
# Example 2: all_matches = csv_to_matches('path/to/csv_file.csv')
#            matches_to_run = return_matches_to_run(all_matches, output_folder)
img_in = convert_path(r"T:\archives\lung\dedication\images\HE\DED-ELK-018_HE_E002.tif")
mask_in = convert_path(r"T:\archives\lung\dedication\tissue_masks\HE\DED-ELK-018_HE_E002_tissue_mask.tif")
matches_to_run = [(img_in, mask_in),]

rerun_unfinished = True # [READ THIS -->] this will check if the '<stem>_runtime.txt file is present in the outputfolder. If it is not it means that this file is not processed fully. IMPORTANT: if this is set to True multiple parallel jobs cannot track which slides are being processed, set this to false if you want to run multiple jobs processing the same 'matches_to_run' in a parralel. Alternatively, remove all unfinished files in the output folder (the files that do not have a .runtime.txt file). Default is False 
overwrite = True # [READ THIS -->] this will always process and overwrite files, even if they are already processed. May break logic if you have multiple jobs processing the same 'matches_to_run' running in parallel. Default is False

### SAMPLING STUFF
spacing = 0.5 # spacing for batch_shape spacing and annotation_parser output_spacing (leave its processing spacing on 4 or higher)
model_patch_size = 512 # input size of model (should be square)
sampler_patch_size = 4 * model_patch_size # use this as batch shape (= 8 * model_patch_size)
cpus = 1 # number of cpus to use for sampling (note that buffer states are printed during inference time now, which may give you information about the number of cpus you should use, I noticed 1 is already enough to have a saturated buffer state when you sample: 4 * model_patch_size)

### WANDB
use_wandb = False # If True put "export WANDB_API_KEY=<your key>"" in your .bashrc or .bash_profile

#################################################################
### USAGE
#################################################################

"""
1) Provide only the stem of the config file
    python3 -u /data/temporary/joey/github/nnUNet-for-pathology_v1/nnunet_for_pathology_example_usage/nnUNet_run_WSI_inference_REWORK_using_config.py example_config
OR
2) Provide the stem of the config file and the input and output paths
    python3 -u /data/temporary/joey/github/nnUNet-for-pathology_v1/nnunet_for_pathology_example_usage/nnUNet_run_WSI_inference_REWORK_using_config.py example_config /data/temporary/archives/lung/dedication/images/HE/DED-ELK-018_HE_E002.tif /data/temporary/archives/lung/dedication/tissue_masks/HE/DED-ELK-018_HE_E002_tissue_mask.tif /data/temporary/joey/data/nnunet_v1_test/DED-ELK-018_HE_E002_nnunet.tif /data/temporary/joey/data/nnunet_v1_test/DED-ELK-018_HE_E002_nnunet.tif/data/temporary/joey/data/nnunet_v1_test/DED-ELK-018_HE_E002_uncertainty.tif
"""