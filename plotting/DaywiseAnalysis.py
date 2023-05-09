import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.ndimage
from matplotlib.colors import LogNorm
import matplotlib.cm as cm


class DaywiseAnalysis:
    """
    A class for plotting daywise histograms and boxplots of fish movement data.

    Attributes:
        df (pandas.DataFrame): The DataFrame containing the daywise analysis data.
        histogram_file_positions (list): A list of file paths to the histogram files.
        fishID (list): A list of tuples where each tuple contains fishID and tanknumber.
        hists (numpy.ndarray): A 4D numpy array of histograms.

    Methods:
        run_analysis(): Loads the normalized histograms, sorts them by sex, calculates the median histograms, 
                        creates daywise histogram plots for male and female fish, and generates a boxplot.
    """
    
    def __init__(self,df):
        """
        Initialize the Back class with the parent_directory and tag.

        Args:
            df (pandas.DataFrame): The DataFrame containing the daywise analysis data.
        """
        self.df = df
        self.histogram_file_positions = self.find_npy_files(self.parent_directory)
        self.fishID, self.hists = self.load_normed_histograms(self.histogram_file_positions)


    def get_day_data(self, data_3d, day):
        """
        Get the data for a specific day from a 3D array.

        Args:
            data_3d (numpy.ndarray): A 3D numpy array containing data for multiple days.
            day (int): The day for which the data is required.

        Returns:
            numpy.ndarray: A 2D numpy array containing data for the specified day.
        """

        return data_3d[day - 1]

    def plot_histogram(self, ax, data, cmap, norm, day):
        """
        Plot a heatmap of the data on the given axis.

        Args:
            ax (matplotlib.axes.Axes): The axis to plot the heatmap on.
            data (numpy.ndarray): A 2D numpy array containing the data to plot.
            cmap (str): The colormap to use for the heatmap.
            norm (matplotlib.colors.Normalize): The normalization used for scaling data to colormap.
            day (int): The day number to be displayed in the plot title.
        """
        data_smooth = scipy.ndimage.zoom(data, 3)
        sns.heatmap(
            data=data_smooth,
            cmap=cmap,
            norm=norm,
            ax=ax,
        )
        ax.set_title(f'Day {day}')
        ax.set_axis_off()

    def create_daywise_histograms(self, data_3d):
        """
        Create a 4x6 grid of heatmaps for each day's histogram using the provided 3D data.

        Args:
            data_3d (numpy.ndarray): A 3D numpy array containing data for multiple days.

        Returns:
            tuple: A tuple containing:
                - matplotlib.figure.Figure: The figure containing the 4x6 grid of heatmaps.
                - matplotlib.figure.Figure: The figure containing the colorbar for the heatmaps.
        """
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
                self.plot_histogram(ax, day_data, cmap, norm, day+1)
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


    def create_vertical_box_stripplot(self, x_col, y_col, hue_col=None, hue_order=None):
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
        sns.boxplot(x=x_col, y=y_col, hue=hue_col, hue_order=hue_order, data=self.df, whis=[0, 100], width=.6, palette="vlag")
        
        # Add in points to show each observation
        sns.stripplot(x=x_col, y=y_col, hue=hue_col, hue_order=hue_order, data=self.df, size=4, color=".3", linewidth=0)
        
        # Tweak the visual presentation
        ax.yaxis.grid(True)
        ax.set(xlabel="")
        sns.despine(trim=True, left=True)
        
        plt.show()

    def find_npy_files(self,directory):
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

    def extract_fishID_tanknumber(self, file_path):
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

    def load_npy_file(self, file_path):
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

    def normalise_histograms(self, histogram):
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
        

    def adjust_histogram_shape(self, hist, max_days):
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

    def load_normed_histograms(self,histogram_file_positions, max_days=22):
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

            hist = self.load_npy_file(file_position)
            hist = self.normalise_histograms(hist)
            hist = self.adjust_histogram_shape(hist, max_days) # Adjust the histogram shape
            fishID = self.extract_fishID_tanknumber(file_position)

            fishes.append(fishID)
            histograms.append(hist)

        histograms = np.stack(histograms, axis=3)
        return fishes, histograms

    def sort_hists_by_sex(self, fishID):
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
        for i, row in self.df.iterrows():
            sex_map[(row['Tank_number'], row['ID'])] = row['Sex']

        # Initialize empty arrays for male and female histograms
        male_hists   =list()
        female_hists = list()

        # Sort the histograms into male and female arrays based on the sex of the fish
        for i, (tank_num, fish_id) in enumerate(fishID):
            sex = sex_map.get((tank_num, fish_id))
            if sex == 'M':
                male_hists.append(self.hists[:,:,:,i])
            elif sex == 'F':
                female_hists.append(self.hists[:,:,:,i])

        male_hists = np.stack(male_hists, axis=3)
        female_hists = np.stack(female_hists, axis=3)

        return male_hists, female_hists