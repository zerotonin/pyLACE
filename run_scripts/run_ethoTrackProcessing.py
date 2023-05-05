from data_base_analyser.EthoVisionExperimentSeries import EthoVisionExperimentSeries
import os
import pandas as pd
import numpy as np
import seaborn as sns
import glob
import matplotlib.pyplot as plt
import scipy.ndimage
from matplotlib.colors import LogNorm
import matplotlib.cm as cm

def get_day_data(data_3d, day):
    return data_3d[day - 1]

def plot_histogram(ax, data, cmap, norm, day):
    data_smooth = scipy.ndimage.zoom(data, 3)
    sns.heatmap(
        data=data_smooth,
        cmap=cmap,
        norm=norm,
        ax=ax,
    )
    ax.set_title(f'Day {day}')
    ax.set_axis_off()

def create_daywise_histograms(data_3d):
    """Create a 4x6 grid of plots for each day's histogram."""
    # Calculate global color axis limits
    vmin, vmax = np.nanmin(data_3d[data_3d > 0]), np.nanmax(data_3d)
    
    # Create a colormap
    cmap = "viridis"
    norm = LogNorm(vmin=vmin if vmin > 0 else 0.01, vmax=vmax)  # Ensure vmin is strictly positive
    
    # Create a 4x6 subplot grid
    fig, axes = plt.subplots(4, 6, figsize=(24, 16), sharex=True, sharey=True, facecolor='white')
    axes = axes.flatten()  # Flatten the 2D list to 1D for easier iteration
    
    # Set a dark background
    plt.style.use('dark_background')
    
    # Plot histograms
    for day, ax in enumerate(axes):
        if day < data_3d.shape[0]:  # Check that we have data for this day
            day_data = data_3d[day]
            plot_histogram(ax, day_data, cmap, norm, day+1)
            if day == 18:  # For 19th plot (index 18), add labels
                ax.set_xlabel('X (cm)')
                ax.set_ylabel('Y (cm)')
        else:
            ax.axis('off')  # If there's no data for this day, hide the axis

      # Create a colorbar in a separate figure
    fig_cbar = plt.figure(figsize=(3, 8), facecolor='white')
    cbar_ax = fig_cbar.add_axes([0.1, 0.2, 0.3, 0.6])
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=14, colors='black')
    return fig,fig_cbar


def create_vertical_box_stripplot(df, x_col, y_col, hue_col=None, hue_order=None):
    """
    Creates a vertical box and strip plot for the specified DataFrame and columns.

    Args:
        df (pandas.DataFrame): The DataFrame to use for plotting.
        x_col (str): The name of the column to use as the x-axis.
        y_col (str): The name of the column to use as the y-axis.
        hue_col (str, optional): The name of the column to use for the hue. Defaults to None.
        hue_order (list, optional): The order to use for the hue categories. Defaults to None.
    """
    sns.set_theme(style="ticks")

    # Initialize the figure with a logarithmic y axis
    f, ax = plt.subplots(figsize=(7, 6))
    #ax.set_yscale("log")

    # Plot the data with vertical boxes
    sns.boxplot(x=x_col, y=y_col, hue=hue_col, hue_order=hue_order, data=df, whis=[0, 100], width=.6, palette="vlag")
    
    # Add in points to show each observation
    sns.stripplot(x=x_col, y=y_col, hue=hue_col, hue_order=hue_order, data=df, size=4, color=".3", linewidth=0)
    
    # Tweak the visual presentation
    ax.yaxis.grid(True)
    ax.set(xlabel="")
    sns.despine(trim=True, left=True)
    
    plt.show()



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

def load_normed_histograms(histogram_file_positions, max_days=22):
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

def sort_hists_by_sex(hists, fishID, df):
    """
    Sorts histograms into two arrays for male and female fish based on the sex information in the DataFrame.

    Args:
        hists (numpy.ndarray): A 4D numpy array of histograms.
        fishID (list): A list of tuples where each tuple contains fishID and tanknumber.
        df (pandas.DataFrame): A DataFrame containing the sex information for each fish.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: A 4D numpy array of histograms for male fish.
            - numpy.ndarray: A 4D numpy array of histograms for female fish.
    """
    # Map each fish ID and tank number to its sex using the DataFrame
    sex_map = {}
    for i, row in df.iterrows():
        sex_map[(row['Tank_number'], row['ID'])] = row['Sex']

    # Initialize empty arrays for male and female histograms
    male_hists   =list()
    female_hists = list()

    # Sort the histograms into male and female arrays based on the sex of the fish
    for i, (tank_num, fish_id) in enumerate(fishID):
        sex = sex_map.get((tank_num, fish_id))
        if sex == 'M':
            male_hists.append(hists[:,:,:,i])
        elif sex == 'F':
            female_hists.append(hists[:,:,:,i])

    male_hists = np.stack(male_hists, axis=3)
    female_hists = np.stack(female_hists, axis=3)

    return male_hists, female_hists

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


male_hists,female_hists = sort_hists_by_sex(hists,fishID,df)
male_hists = normalise_histograms(np.nanmedian(male_hists,axis=3))
female_hists = normalise_histograms(np.nanmedian(female_hists,axis=3))

create_daywise_histograms(male_hists)
create_daywise_histograms(female_hists)

for topic in ['Median_speed_cmPs', 'Gross_speed_cmPs',
       'Median_activity_duration_s', 'Activity_fraction',
       'Median_freezing_duration_s', 'Freezing_fraction',
       'Median_top_duration_s', 'Top_fraction', 'Median_bottom_duration_s',
       'Bottom_fraction', 'Median_tigmotaxis_duration_s',
       'Tigmotaxis_fraction', 'Tigmotaxis_transitions', 'Latency_to_top_s',
       'Distance_travelled_cm']:
    
    create_vertical_box_stripplot(df,'Day_number',topic,'Sex',('M','F'))
    plt.show()


create_vertical_box_stripplot(df,'Day_number','Top_fraction','Sex',('M','F'))
plt.show()



def create_top_fraction_lineplot(df,measure):
    """
    Creates a line plot for the Top_fraction over Day_number for each fish separately,
    with individual males in blue and females in red, based on the Tank_number, ID, and Sex columns in the DataFrame df.

    Args:
        df (pandas.DataFrame): The DataFrame to use for plotting.
    """
    # Group the data by Tank_number, ID, and Sex
    groups = df.groupby(["Tank_number", "ID", "Sex"])

    # Loop through the groups and plot the lines with the corresponding colors and markers
    for (tank_num, fish_id, sex), group in groups:
        color = "blue" if sex == "M" else "red"
        marker = "o" if group[measure].iloc[0] >= 0.4 else "x"
        label = f"{tank_num}, {fish_id}"

        # Create the plot
        fig, ax = plt.subplots()
        ax.plot(group["Day_number"], group[measure], color=color, label=label, marker=marker)

        # Add labels and title
        ax.set_xlabel("Day number")
        ax.set_ylabel(measure)
        ax.set_title(f"Fish ID: {fish_id}, Tank number: {tank_num}")
        ax.set_ylim((0,500))

        # Add legend and markers
        ax.legend()
        ax.axhline(y=0.4, color="gray", linestyle="--")
        ax.plot([], [], color=color, marker=marker, label="Fraction above 0.4")

        # Show the plot
        plt.show()

create_top_fraction_lineplot(df,'Tigmotaxis_transitions')