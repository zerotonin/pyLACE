from data_base_analyser.EthoVisionExperimentSeries import EthoVisionExperimentSeries
import os
import pandas as pd
import numpy as np
import seaborn as sns
import os
import glob

def find_npy_files(directory):
    """
    Find all npy files in a directory including its subdirectories.

    Args:
        directory (str): The path to the directory to search.

    Returns:
        file_list (list): A list of paths to the npy files.
    """
    file_list = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for file in glob.glob(os.path.join(dirpath, "*.npy")):
            file_list.append(file)
    return file_list

def extract_fishID_tanknumber(file_path):
    """
    Extract fishID and tanknumber from the file path.

    Args:
        file_path (str): The file path containing fishID and tanknumber.

    Returns:
        tuple: A tuple containing fishID (integer) and tanknumber (string).
    """
    # Extract the directory containing fishID and tanknumber from the file path
    directory = os.path.dirname(file_path)

    # Extract fishID and tanknumber from the directory name
    parts = directory.split(os.path.sep)[-1].split('__')
    tanknumber = int(parts[0].replace('tankNum_', ''))
    fishID = parts[1].replace('fishID_', '')

    return tanknumber, fishID

def load_npy_file(file_path):
    """
    Load the data from a .npy file.

    Args:
        file_path (str): The file path to the .npy file.

    Returns:
        numpy.ndarray: The numpy array containing the data from the .npy file.
    """
    return np.load(file_path)

def normalise_histograms(histogram):
    return histogram/histogram.sum(axis=(1,2), keepdims=True)

def load_normed_histograms(histogram_file_positions,max_days=21):
    fishes = list()
    histograms = list()
    for file_position in histogram_file_positions:   

        hist = load_npy_file(file_position)
        hist = normalise_histograms(hist)
        fishID = extract_fishID_tanknumber(file_position)

        fishes.append(fishID)
        histograms.append(hist)

    histograms = np.stack(histograms,axis=3)
    return fishes,histograms

#[print(x.shape) for x in histograms]
# Usage
tag = 'habituation2023'
parent_directory = '/home/bgeurten/ethoVision_database/'

# Compile data daywise
#etho_vision_analysis = EthoVisionExperimentSeries(tag, parent_directory)
#etho_vision_analysis.process_and_save()

db_position = f'{parent_directory}{tag}_daywise_analysis.csv'
df = pd.read_csv(db_position)

histogram_file_positions = find_npy_files(parent_directory)

fishID, hists = load_normed_histograms(histogram_file_positions)