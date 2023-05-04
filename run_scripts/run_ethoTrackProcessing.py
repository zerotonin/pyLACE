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

    Raises:
        FileNotFoundError: If the .npy file does not exist.
        ValueError: If the file_path is not a string.
    """
    if not isinstance(file_path, str):
        raise ValueError("The file_path should be a string.")

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"No .npy file found at {file_path}")

    return np.load(file_path)

def normalise_histograms(histogram):
    """
    Normalises the provided histogram.

    The normalisation is performed over the second and third axes of the histogram. 
    It divides each value in the histogram by the sum of all values in the same 2D slice.

    Args:
        histogram (numpy.ndarray): A 3D numpy array representing the histogram to be normalised.

    Returns:
        numpy.ndarray: The normalised histogram.

    Raises:
        ValueError: If the input is not a 3D numpy array.
    """
    if not isinstance(histogram, np.ndarray) or histogram.ndim != 3:
        raise ValueError("The input histogram should be a 3D numpy array.")

    return histogram / histogram.sum(axis=(1, 2), keepdims=True)
    return histogram/histogram.sum(axis=(1,2), keepdims=True)

import numpy as np

def adjust_histogram_shape(hist, max_days):
    """
    Adjust the shape of the histogram in the first axis to match the max_days.
    If there are more than max_days entries, drop the first entries.
    If there are less than max_days entries, fill the rest with np.nan.

    Args:
        hist (numpy.ndarray): The input histogram.
        max_days (int): The maximum number of days.

    Returns:
        numpy.ndarray: The adjusted histogram.
    """
    hist_days = hist.shape[0]

    if hist_days > max_days:
        return hist[-max_days:]
    elif hist_days < max_days:
        padding_shape = (max_days - hist_days,) + hist.shape[1:]
        padding = np.full(padding_shape, np.nan)
        return np.concatenate((padding, hist), axis=0)
    else:
        return hist

def load_normed_histograms(histogram_file_positions, max_days=21):
    """
    Loads the normalized histograms from the provided file positions. 

    The histogram data is adjusted according to the max_days parameter. 
    If there are more than max_days entries, the first entries are dropped. 
    If there are less than max_days entries, the rest are filled up with np.nan.

    Args:
        histogram_file_positions (list): List of file paths to the histogram files.
        max_days (int, optional): The maximum number of days for which the data 
                                   should be loaded. Defaults to 21.

    Returns:
        tuple: A tuple containing:
            - list: A list of tuples where each tuple contains fishID and tanknumber.
            - numpy.ndarray: A 4D numpy array of histograms, where the first axis 
                             represents the day, and the last axis represents the different fish.

    Raises:
        FileNotFoundError: If the file in one of the histogram_file_positions does not exist.
    """
    fishes = list()
    histograms = list()
    for file_position in histogram_file_positions:   

        hist = load_npy_file(file_position)
        hist = normalise_histograms(hist)
        hist = adjust_histogram_shape(hist, max_days) # Adjust the histogram shape
        fishID = extract_fishID_tanknumber(file_position)

        fishes.append(fishID)
        histograms.append(hist)

    histograms = np.stack(histograms, axis=3)
    return fishes, histograms


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