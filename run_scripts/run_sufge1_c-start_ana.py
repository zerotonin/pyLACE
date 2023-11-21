import pandas as pd
import fish_data_base.fishDataBase as fishDataBase
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import re, os
import dill as pickle
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from scipy.signal import butter, filtfilt
from scipy.stats import norm
import seaborn as sns
from scipy.stats import sem
from scipy.ndimage import gaussian_filter1d

def extract_info_from_id_text(identifier_text):
    """
    Extract relevant information from an identifier text.
    
    Given an identifier text, this function extracts the strain type, sex, 
    and fish number based on pre-defined mappings and string patterns.
    
    Args:
        identifier_text (str): The identifier text to be parsed.
        
    Returns:
        tuple: A tuple containing the strain type (str), sex (str), and fish number (int).
    
    Example:
        >>> extract_info_from_id_text("sample_Hm123M4IIII")
        ('sufge1-HM', 'M', 4)
    """
    
    # Mapping of short strain identifiers to full identifiers
    strain_map = {'Ht':'sufge1-HT', 'Hm':'sufge1-HM', 'Int':'sufge1-INT'}
    
    # Extract the identifier part from the input string
    id = identifier_text.split('_')[1]
    
    # Determine the strain and the rest of the ID based on the pattern
    if id[1] == 'n':
        strain = id[:3]
        rest = id[3:]
    else:
        strain = id[:2]
        rest = id[2:]
        
    # Map the short strain identifier to the full identifier
    strain = strain_map[strain]
    
    # Extract the sex from the rest of the ID
    sex = rest[:1]
    rest = rest[1:]
    
    # Extract the fish number using regex to find digits
    fish_no = int(re.findall(r'\d+', rest)[0])
    
    return strain, sex, fish_no


def get_val(df,field):
    """
    Retrieve the first value in a specified DataFrame column.
    
    Args:
        df (pd.DataFrame): The DataFrame from which to retrieve the value.
        field (str): The column name.
        
    Returns:
        any: The first value in the specified column.
    """
    return  df[field].iloc[0]



def calculate_vector_norms(midline_df):
    """
    Calculate the sum of vector norms for each row in a DataFrame.

    Each row in the DataFrame represents a series of coordinates, and this function calculates
    the sum of the Euclidean distances between each consecutive pair of points in a row.

    Args:
        midline_df (pd.DataFrame): DataFrame containing coordinates.

    Returns:
        list: A list of sums of vector norms for each row in the DataFrame.
    """
    vector_norm_sums = []
    
    # Iterate through each row in the DataFrame
    for index, row in midline_df.iterrows():
        norm_sum = 0
        
        # Iterate through each pair of consecutive points
        for i in range(0, 9):
            x1, y1 = row[f'x_coord_{i}'], row[f'y_coord_{i}']
            x2, y2 = row[f'x_coord_{i+1}'], row[f'y_coord_{i+1}']
            
            # Calculate the Euclidean distance between the two points
            vector_norm = np.linalg.norm([x2 - x1, y2 - y1])
            
            # Accumulate the sum of vector norms
            norm_sum += vector_norm
        
        # Append the sum of vector norms for the current row to the list
        vector_norm_sums.append(norm_sum)
    
    return vector_norm_sums


def filter_by_criteria(df, strain, sex, fish_no):
    """
    Filter a DataFrame based on specified criteria for strain, sex, and fish number.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        strain (str): The strain type to filter by.
        sex (str): The sex to filter by.
        fish_no (int): The fish number to filter by.

    Returns:
        pd.DataFrame: A filtered DataFrame based on the given criteria.
    """
    return df.loc[(df['genotype'] == strain) & (df['sex'] == sex) & (df['animalNo'] == fish_no), :]

def check_paths_available(df, path_keys):
    """
    Check if specified paths are available in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing path information.
        path_keys (list of str): List of column names to check for paths.

    Returns:
        bool: True if all paths are available, False otherwise.
    """
    for path_key in path_keys:
        path = get_val(df, path_key)
        if pd.isna(path):
            print(f"Skipping {path_key} as it's not available")
            return False
    return True

def calculate_head_tail_norms(midline_df):
    """
    Calculate the Euclidean distances between the head and tail coordinates for each row in a DataFrame.

    Args:
        midline_df (pd.DataFrame): DataFrame containing head and tail coordinates.

    Returns:
        list: A list of head-tail Euclidean distances for each row in the DataFrame.
    """
    vector_norm_sums = []
    
    # Iterate through each row in the DataFrame
    for index, row in midline_df.iterrows():
        

        x1, y1 = row[f'x_coord_0'], row[f'y_coord_0']
        x2, y2 = row[f'x_coord_9'], row[f'y_coord_9']
        
        # Calculate the Euclidean distance between the two points
        norm_sum = np.linalg.norm([x2 - x1, y2 - y1])
        
        # Accumulate the sum of vector norms
           
        
        # Append the sum of vector norms for the current row to the list
        vector_norm_sums.append(norm_sum)
    
    return vector_norm_sums

def calculate_tortuosity(midline_df):
    """
    Calculate tortuosity based on the sum of vector norms and head-tail distance for each row in a DataFrame.

    Args:
        midline_df (pd.DataFrame): DataFrame containing midline coordinates.

    Returns:
        pd.DataFrame: Updated DataFrame with a new column 'tortuosity'.
    """

    # Calculate the sum of vector norms for each row and store it in a new DataFrame column
    midline_df['vector_norm_sum'] = calculate_vector_norms(midline_df)
    midline_df['head_tail_dist'] = calculate_head_tail_norms(midline_df)

    midline_df['tortuosity'] = (midline_df['vector_norm_sum'] - midline_df['head_tail_dist'])/midline_df['vector_norm_sum'] 
    return midline_df


def find_tortuosity_peaks(midline_df, threshold=0.66):
    """
    Identify peaks in tortuosity above a specified threshold.

    Args:
        midline_df (pd.DataFrame): DataFrame containing tortuosity data.
        threshold (float): The threshold value to identify peaks.

    Returns:
        tuple: Indices of peaks, corresponding time in seconds, and tortuosity values at peaks.
    """
    # Extract the tortuosity values from the DataFrame
    tortuosity_values = midline_df['tortuosity'].to_numpy()

    # Use find_peaks to find indices of peaks based on the threshold
    peaks, _ = find_peaks(tortuosity_values, height=threshold)
    
    return  peaks, midline_df['time sec'].iloc[peaks].to_numpy(), tortuosity_values[peaks]


def interpolate_spike_frequencies(spike_train_df, time_range=(0, 8), resolution=0.001):
    """
    Interpolate spike frequencies over a specified time range.

    This function interpolates the spike frequencies of a fish over a given time range 
    with a specified resolution, using the spike peak times and instantaneous frequencies 
    recorded in the spike_train_df DataFrame.

    Args:
        spike_train_df (pd.DataFrame): DataFrame containing spike peak times ('spike_peak_s') 
                                       and instantaneous frequencies ('instant_freq').
        time_range (tuple): A tuple (start, end) defining the time range for interpolation.
        resolution (float): The time resolution for interpolation.

    Returns:
        pd.DataFrame: A DataFrame containing the interpolated spike frequencies 
                      over the specified time range.
    """
    time_grid = np.arange(time_range[0], time_range[1], resolution)
    interpolated_frequencies = np.interp(time_grid, spike_train_df['spike_peak_s'] - 1, spike_train_df['instant_freq'], left=0, right=0)
    return pd.DataFrame({'time_sec': time_grid, 'interpolated_instant_freq': interpolated_frequencies})

def read_csv(fish_df,path_field):
    """
    Read a CSV file specified by a path in a DataFrame field.

    Args:
        fish_df (pd.DataFrame): DataFrame containing the path information.
        path_field (str): The column name in fish_df that contains the path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame read from the specified CSV file.
    """
    # Retrieve the path to the CSV file from the specified field
    path = get_val(fish_df, path_field)

    # Read and return the CSV file as a DataFrame
    return pd.read_csv(path)

def read_all_csv(fish_df):
    """
    Read multiple CSV files specified in a DataFrame.

    This function reads three specific CSV files (spike train data, midline data, and trace data)
    whose paths are specified in given fields of the fish_df DataFrame.

    Args:
        fish_df (pd.DataFrame): DataFrame containing the path information for each CSV file.

    Returns:
        tuple: A tuple of DataFrames read from the specified CSV files (spike_train_df, midline_df, trace_df).
    """
    # Read each CSV file specified in the given fields of fish_df
    spike_train_df = read_csv(fish_df, 'path2_spike_train_df')
    midline_df = read_csv(fish_df, 'path2_midLineUniform_pix')
    trace_df = read_csv(fish_df, 'path2_trace_mm')

    return spike_train_df, midline_df, trace_df

def get_tortuosity(midline_df, cutoff_freq=None, sampling_rate=None):
    """
    Calculate tortuosity and optionally apply a low-pass filter.

    This function calculates the tortuosity of a fish based on midline data. It also provides
    the option to apply a low-pass filter to the tortuosity data using either a Butterworth 
    or Gaussian filter, depending on the provided parameters.

    Args:
        midline_df (pd.DataFrame): DataFrame containing midline data.
        cutoff_freq (float, optional): The cutoff frequency for the low-pass filter.
        sampling_rate (float, optional): The sampling rate, required if using Butterworth filter.

    Returns:
        tuple: A tuple containing time in seconds and tortuosity values.
    """
    midline_df = calculate_tortuosity(midline_df)

    # Apply low-pass filter if cutoff frequency and sampling rate are provided
    if cutoff_freq is not None and sampling_rate is not None:
        midline_df['tortuosity'] = butter_lowpass_filter(midline_df['tortuosity'], cutoff_freq, sampling_rate)
    elif cutoff_freq is not None and sampling_rate is None:
        midline_df['tortuosity'] = gaussian_lowpass_filter(midline_df['tortuosity'], sigma=cutoff_freq)
    
    return (midline_df['time sec'], midline_df['tortuosity'])

def get_speed(midline_df,trace_df, cutoff_freq=None, sampling_rate=None):
    """
    Calculate the speed of a fish and optionally apply a low-pass filter.

    This function calculates the translational speed of a fish from trace data. It also
    provides the option to apply a low-pass filter to the speed data.

    Args:
        midline_df (pd.DataFrame): DataFrame containing midline data.
        trace_df (pd.DataFrame): DataFrame containing trace data.
        cutoff_freq (float, optional): The cutoff frequency for the low-pass filter.
        sampling_rate (float, optional): The sampling rate, required if using Butterworth filter.

    Returns:
        tuple: A tuple containing time in seconds and translational speed values in meters per second.
    """
    trace_df.interpolate(inplace=True)
    trace_df['trans_speed_mPs'] = trace_df['thrust_m/s'].abs() + trace_df['slip_m/s'].abs()
    trace_df['trans_speed_mPs'] = trace_df['trans_speed_mPs']/10000

    # Apply low-pass filter if cutoff frequency and sampling rate are provided
    if cutoff_freq is not None and sampling_rate is not None:
        trace_df['trans_speed_mPs'] = butter_lowpass_filter(trace_df['trans_speed_mPs'], cutoff_freq, sampling_rate)
    elif cutoff_freq is not None and sampling_rate is None:
        trace_df['trans_speed_mPs'] = gaussian_lowpass_filter(trace_df['trans_speed_mPs'], sigma=cutoff_freq)
    
    return (midline_df['time sec'],trace_df['trans_speed_mPs'])

def get_spike_freq(spike_train_df, cutoff_freq=None, sampling_rate=None):
    """
    Calculate the speed of a fish and optionally apply a low-pass filter.

    This function calculates the translational speed of a fish from trace data. It also
    provides the option to apply a low-pass filter to the speed data.

    Args:
        midline_df (pd.DataFrame): DataFrame containing midline data.
        trace_df (pd.DataFrame): DataFrame containing trace data.
        cutoff_freq (float, optional): The cutoff frequency for the low-pass filter.
        sampling_rate (float, optional): The sampling rate, required if using Butterworth filter.

    Returns:
        tuple: A tuple containing time in seconds and translational speed values in meters per second.
    """
    spike_freq_df = interpolate_spike_frequencies(spike_train_df)
    time_series = spike_freq_df['time_sec']
    freq_series = spike_freq_df['interpolated_instant_freq']

    # Apply low-pass filter if cutoff frequency and sampling rate are provided
    if cutoff_freq is not None and sampling_rate is not None:
        freq_series = butter_lowpass_filter(freq_series, cutoff_freq, sampling_rate)
    elif cutoff_freq is not None and sampling_rate is None:
        freq_series = gaussian_lowpass_filter(freq_series, sigma=cutoff_freq)


    return (time_series, freq_series)


def get_spike_mauthnerHistogram(spike_train_df):
    """
    Calculate a normalized histogram for Mauthner spike times.

    This function identifies the Mauthner spikes in the spike train data and calculates
    a normalized histogram of their occurrence times.

    Args:
        spike_train_df (pd.DataFrame): DataFrame containing spike train data, 
                                       specifically the 'spike_category' and 'spike_peak_s' columns.

    Returns:
        ndarray: An array representing the normalized histogram of Mauthner spike times.
    """
    mauthner_idx = spike_train_df[spike_train_df['spike_category'] == 'Mauthner'].index
    mauthner_times = spike_train_df['spike_peak_s'].iloc[mauthner_idx] - 1  # Adjusting for the one-second difference
    mauthner_hist = calc_normalised_histogram(mauthner_times.to_numpy())
    return mauthner_hist

def calc_normalised_histogram(events,bins=np.arange(0, 5.1, 0.01)):
    """
    Calculate a normalized histogram for a given set of events.

    This function computes a histogram of the provided events and normalizes it. 
    It is particularly useful for normalizing distributions of event occurrences, 
    such as spike times. NaN values in the histogram (resulting from bins with no events) 
    are replaced with zeros, which is important for cases like fish with no Mauthner spikes.

    Args:
        events (array-like): An array of event occurrences, typically as times or similar measurements.
        bins (array-like, optional): The bin edges, including the rightmost edge, allowing for 
                                     custom bin ranges. Default is set to range from 0 to 5.1 with a 
                                     step of 0.01.

    Returns:
        np.ndarray: A normalized histogram as a NumPy array, where each bin value is a fraction 
                    of the total number of events.
    """
    hist_data = np.histogram(events, bins)
    hist_data = hist_data[0] / hist_data[0].sum(axis=0, keepdims=True)  # Normalize each individual histogram
    hist_data[np.isnan(hist_data)] = 0 # replace nans with zeros, as zeros come from fish that had no Mauthner spikes
    return hist_data

def butter_lowpass_filter(data, cutoff_freq, sampling_rate, order=4):
    """
    Apply a low-pass Butterworth filter to the data.

    Parameters:
    - data: array-like, the data to be filtered
    - cutoff_freq: float, the cutoff frequency of the filter
    - sampling_rate: float, the sampling rate of the data
    - order: int, the order of the filter (default is 4)

    Returns:
    - y: array, the filtered data
    """
    # Calculate the Nyquist frequency
    nyquist_freq = 0.5 * sampling_rate
    
    # Normalize the cutoff frequency by the Nyquist frequency
    normal_cutoff = cutoff_freq / nyquist_freq
    
    # Design the filter
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # Apply the filter to the data using filtfilt
    y = filtfilt(b, a, data)
    
    return y

def gaussian_lowpass_filter(data, sigma):
    """
    Apply a Gaussian low-pass filter to the data.

    Parameters:
    - data: array-like, the data to be filtered
    - sigma: float, the standard deviation for the Gaussian kernel

    Returns:
    - filtered_data: array, the filtered data
    """
    # Apply Gaussian filter
    filtered_data = gaussian_filter1d(data, sigma)
    
    return filtered_data


def collect_data():
    # Assuming fishDataBase and other necessary modules are imported
    db = fishDataBase.fishDataBase("/home/bgeurten/fishDataBase",'/home/bgeurten/fishDataBase/fishDataBase_cstart.csv')
    df = db.database
    df_jump = pd.read_csv('/home/bgeurten/fishDataBase/suf_cstart_round_robin_jumps.csv')

    phenotypes = ['sufge1-HT', 'sufge1-HM', 'sufge1-INT']
    sexes = ['M', 'F']

    data_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    total_rows = len(df_jump)
    for i, row in tqdm(df_jump.iterrows(), total=total_rows, desc='Round robin update'):
           
        genotype, sex , fish_no = extract_info_from_id_text(row[1])
        # Use filter_by_criteria function to filter DataFrame
        fish_df = filter_by_criteria(df, genotype, sex, fish_no)

        if fish_df is not None and len(fish_df) == 1:
            path_keys = ['path2_midLineUniform_pix', 'path2_spike_train_df', 'path2_trace_mm']
            if check_paths_available(fish_df, path_keys):

                # Read all files
                trace_df, midline_df, spike_train_df =read_all_csv(fish_df)
                
                # Sort data directly into data_dict
                data_dict[sex][genotype]['spike_frequency'].append(get_spike_freq(spike_train_df,10))
                data_dict[sex][genotype]['speed'].append(get_speed(midline_df, trace_df,10))
                data_dict[sex][genotype]['tortuosity'].append(get_tortuosity(midline_df,10))
                data_dict[sex][genotype]['mauthner_histogram'].append(get_spike_mauthnerHistogram(spike_train_df))

    return data_dict


def replace_with_summed_histograms(data_dict):
    for sex, sex_data in data_dict.items():
        for genotype, genotype_data in sex_data.items():
            all_histograms = genotype_data['mauthner_histogram']
            # Convert the list of histograms to a NumPy array for easy manipulation
            all_histograms_array = np.array(all_histograms)

            # Sum along axis 0 to get the sum of histograms
            summed_histogram = all_histograms_array.sum(axis=0)
            
            # You can also normalize the summed_histogram if needed
            summed_histogram = summed_histogram / summed_histogram.sum(axis=0, keepdims=True)
            
            # Replace the 'mauthner_histogram' entry with the new summed histogram
            genotype_data['mauthner_histogram'] = summed_histogram


def calculate_median_and_CI_for_field(all_tuples):
    z_value = norm.ppf(0.975)  # Z-value for 95% confidence interval

    # Convert list of tuples into a list of dataframes for easy manipulation
    all_dfs = [pd.DataFrame({'time': np.round(tup[0], 3), 'data': tup[1]}) for tup in all_tuples]  # Rounded to the nearest millisecond

    # Concatenate along a new level of index to keep separate entries
    concatenated_df = pd.concat(all_dfs, keys=range(len(all_dfs)))

    # Compute median and confidence intervals at each time point
    median_df = concatenated_df.groupby('time')['data'].median()
    sample_std = concatenated_df.groupby('time')['data'].std()
    sample_size = concatenated_df.groupby('time').size()

    # Calculate the confidence interval for the median
    lower_bound = median_df - z_value * (sample_std / np.sqrt(sample_size))
    upper_bound = median_df + z_value * (sample_std / np.sqrt(sample_size))

    # Create and return a dictionary with median and confidence interval
    return {
        'central': median_df,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }


def calculate_mean_and_SEM_for_field(all_tuples):
    # Convert list of tuples into a list of dataframes for easy manipulation
    #all_dfs = [pd.DataFrame({'time': np.round(tup[0], 4), 'data': tup[1]}) for tup in all_tuples]  # Rounded to the nearest millisecond
    all_dfs = [pd.DataFrame({'time': np.ceil(tup[0] * 1000) / 1000, 'data': tup[1]}) for tup in all_tuples]

    # Concatenate along a new level of index to keep separate entries
    concatenated_df = pd.concat(all_dfs, keys=range(len(all_dfs)))

    # Compute mean and SEM at each time point
    mean_df = concatenated_df.groupby('time')['data'].mean()
    sample_sem = concatenated_df.groupby('time')['data'].apply(sem)
    
    # Calculate the standard error bounds for the mean
    lower_bound = mean_df - sample_sem
    upper_bound = mean_df + sample_sem

    # Create and return a dictionary with mean and SEM
    return {
        'central': mean_df,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }

# Example usage
# all_tuples is a list where each tuple contains (time_series, corresponding_data)
# e.g., [(time_series1, data1), (time_series2, data2), ...]
# result = calculate_mean_and_SEM_for_field(all_tuples)


def replace_fields_in_dict(data_dict, fields, mode = 'median'):
    for sex, sex_data in data_dict.items():
        for genotype, genotype_data in sex_data.items():
            for field in fields:
                all_tuples = genotype_data[field]
                if mode == 'mean':
                    genotype_data[field] = calculate_mean_and_SEM_for_field(all_tuples)
                elif mode == 'median':
                    genotype_data[field] = calculate_median_and_CI_for_field(all_tuples)

# Set up colour-blind-friendly palette
palette = sns.color_palette("colorblind")

def plot_line_with_shade(ax, data, label, color):
    ax.plot(data['central'], label=label, color=color)
    ax.fill_between(data['central'].index, data['lower_bound'], data['upper_bound'], color=color, alpha=0.3)

def plot_mauthner_bars(ax, data, label, color, off_set = 1.1):
    # Indices to plot (95 to 125)
    start_index = 95
    end_index = 125
    selected_data = data[start_index:end_index]

    # New x-axis scale from -0.15 to 0.15
    new_x_scale = np.linspace(-0.15, 0.15, end_index - start_index)

    # Define the width of each bar (adjust as needed)
    bar_width = (0.30 / (end_index - start_index))  # Width based on the range and number of bars

    # Create the bar plot
    ax.bar(new_x_scale, selected_data, width=bar_width, label=label, color=color, alpha=0.7)

    # Set the x-axis limits
    ax.set_xlim(-0.15, 0.15)
    # x_axis = np.linspace(0,len(data)/100,len(data)+1) - off_set
    # ax.bar(x_axis[0:len(data)], data, label=label, color=color, alpha=0.7)
    # ax.set_xlim((x_axis[95],x_axis[125]))

def plot_individual_field(ax, data_dict, field, plot_function, xlims=None, new_zero=None):
    for idx, (genotype, genotype_data) in enumerate(data_dict.items()):
        color = palette[idx]

        # If new_zero is specified, shift the data
        if new_zero is not None:
            for stats_value in ['central', 'lower_bound', 'upper_bound']:
                genotype_data[field][stats_value].index = genotype_data[field][stats_value].index - new_zero

        plot_function(ax, genotype_data[field], label=genotype, color=color)

    if xlims:
       
        ax.set_xlim((-1*xlims,xlims))
        
    ax.legend()

def plot_data(data_dict):
    for sex, sex_data in data_dict.items():
        fig, axes_2d = plt.subplots(2, 2, figsize=(6, 6), sharey=False)

        # Flatten the 2D array to 1D
        axes = axes_2d.ravel()

        meta_data = (('abs. speed, m/s',(0,1.5)),('tortuosity, norm.',(0,0.2)),('inst. spike frequency, Hz',(0,6000)))
        c = 0

        # Plot speed, tortuosity, and spike frequency as line plots
        for ax, field in zip(axes[:-1], [('speed',0.15,0.725), ('tortuosity',0.15,0.75), ('spike_frequency',0.15,1.1)]):
            plot_individual_field(ax, sex_data, field[0], plot_line_with_shade, field[1], field[2])
            ax.set_title(f'{field[0].capitalize()} in {sex}')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(meta_data[c][0])
            ax.set_ylim(meta_data[c][1])
            c +=1
        
        # Plot mauthner_histogram as bar plot
        plot_individual_field(axes[-1], sex_data, 'mauthner_histogram', plot_mauthner_bars)
        axes[-1].set_title(f'Mauthner cells in {sex}')
        axes[-1].set_xlabel('Time (s)')
        axes[-1].set_ylabel('fraction')

        
        plt.tight_layout()
    plt.show()

def save_data_to_disk(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_data_from_disk(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def main():
    file_path = '/home/bgeurten/fishDataBase/c-start_combined_timeData.pkl'  # Replace with your desired file path
    
    # Check if the data_dict file already exists on disk
    if os.path.exists(file_path):
        print("Loading data from disk...")
        data_dict = load_data_from_disk(file_path)
    else:
        print("Collecting data...")
        data_dict = collect_data()
        print("Saving data to disk...")
        save_data_to_disk(data_dict, file_path)


    replace_with_summed_histograms(data_dict)
    # List of fields to apply the function on
    fields_to_replace = ['speed', 'tortuosity', 'spike_frequency']

    # After populating your data_dict, call this function to replace the individual data points
    replace_fields_in_dict(data_dict, fields_to_replace,'median')

    plot_data(data_dict)

    print("Done")

if __name__ == '__main__':
    main()

