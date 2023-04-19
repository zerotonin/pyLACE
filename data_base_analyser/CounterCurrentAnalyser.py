import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import fish_data_base.fishDataBase as fishDataBase
import re
from scipy.interpolate import interp2d
from mpl_toolkits.axes_grid1 import make_axes_locatable

class CounterCurrentAnalyser:
    """
    A class used to analyze fish data and generate various plots and statistics.
    
    Attributes:
    ----------
    df : pd.DataFrame
        The input DataFrame containing fish data.
        
    Methods:
    -------
    __init__(self, df, genotypes=['rei-INT', 'rei-HM', 'rei-HT'], sexes=['F', 'M'])
        Initializes the CounterCurrentAnalyser with the input DataFrame and filters it based on genotypes and sexes.
        
    filter_dataframe(genotypes, exp, sexes)
        Filters the input DataFrame based on a list of genotypes, an experiment string, and a list of sexes.
        
    generate_grouped_data()
        Generates a dictionary containing 3D numpy arrays for each sex-genotype combination in the input DataFrame.
        
    create_all_histograms(interp_factor=3, clim=(0.0, 0.03))
        Creates 2D histograms with marginals for the given result dictionary and returns a list of figures.
        
    generate_boxplot_data(mode='all')
        Generates a boxplot DataFrame for the given result dictionary based on the specified mode.
        
    plot_boxplot()
        Plots a boxplot for the given DataFrame and returns the created figure.
    
    save_result(boxplot_df, box_fig, hist_fig, fig_path, data_path)
        Saves the boxplot figure, histogram figures, and boxplot data to the specified paths.
        
    main(fig_path, data_path, distance_mode='all', interp_factor=3, clim=(0.0, 0.03))
        Generates and saves all plots and statistics using the specified parameters.
    """

    def __init__(self, df, genotypes=['rei-INT', 'rei-HM', 'rei-HT'], sexes=['F', 'M']):
        self.df = df
        self.df  = self.filter_dataframe(genotypes, 'counter current', sexes)
   
    
    #╔════════════════════════════════════════════════════════════════════════════════╗
    #║                          DATA PROCESSING FUNCTIONS                             ║
    #╚════════════════════════════════════════════════════════════════════════════════╝

    def filter_dataframe(self, genotypes, exp, sexes):
        """
        Filters a dataframe based on a list of genotypes, an experiment string, and a list of sexes.

        Args:
        df (pd.DataFrame): The input dataframe.
        genotypes (list): A list of genotype values to filter by.
        exp (str): The experiment string to filter by.
        sexes (list): A list of sexes to filter by.

        Returns:
        pd.DataFrame: A filtered dataframe containing only rows that match the given conditions.
        """
        filtered_df = self.df[self.df['genotype'].isin(genotypes) &
                        (self.df['expType'] == exp) &
                        self.df['sex'].isin(sexes)]
        return filtered_df




    def extract_numbers_from_columnnames(self,columns):
        """
        Extracts numbers from column names and returns them as a numpy array.

        Parameters:
        columns (iterable): A list or array-like object containing column names.

        Returns:
        numpy.ndarray: A numpy array containing the extracted numbers from the column names.
        """
        numbers = []
        for col in columns:
            # Find all digits in the column name
            number = re.findall('\d+', col)

            # If digits are found, append the first one to the 'numbers' list
            if number:
                numbers.append(int(number[0]))

        # Return the numbers as a numpy array
        return np.array(numbers)
    

    def read_prob_density_csv(self,file_position):
        """
        Reads a probability density CSV file and returns data, x-axis, and y-axis as numpy arrays.

        Parameters:
        file_position (str): The path of the CSV file.

        Returns:
        tuple: A tuple containing three numpy arrays: data, x-axis, and y-axis.
        """
        # Read the CSV file into a pandas DataFrame
        pd_df = pd.read_csv(file_position)

        # Drop the 'Unnamed: 0' column if it exists
        if 'Unnamed: 0' in pd_df.columns:
            pd_df = pd_df.drop(columns='Unnamed: 0')

        # Extract the numbers from the column names to create the x-axis
        x_axis = self.extract_numbers_from_columnnames(pd_df.columns)

        # Get the y-axis values from the 'orthoIndexMM' column
        y_axis = pd_df.orthoIndexMM.values

        # Get the data as a numpy array, excluding the first column (orthoIndexMM)
        data = pd_df.iloc[:, 1:].to_numpy()

        # Normalise histogram
        data = data/np.sum(np.sum(data,axis=1),axis=0)

        # Return the data, x-axis, and y-axis as a tuple
        return (data, x_axis, y_axis)
    
    def generate_grouped_data(self):
        """
        Generates a dictionary containing 3D numpy arrays for each sex-genotype combination in the given DataFrame.

        Returns:
            dict: A dictionary containing 3D numpy arrays for each sex-genotype combination.
        """
        grouped = self.df.groupby(['sex', 'genotype'])
        result = {}

        for (sex, genotype), group in grouped:
            data_arrays = []

            for _, row in group.iterrows():
                data, _, _ = self.read_prob_density_csv(row['path2_probDensity'])
                data_arrays.append(data)

            combined_data = np.stack(data_arrays)
            result[(sex, genotype)] = combined_data

        return result
    
    #╔════════════════════════════════════════════════════════════════════════════════╗
    #║                       DISTANCE CALCULATION FUCNTIONS                           ║
    #╚════════════════════════════════════════════════════════════════════════════════╝

    def calculate_distances_to_core(self,data):
        """
        Calculate the Euclidean distance from the center of each 2D array to the position of the maximum value.
        
        Parameters:
        data (numpy.ndarray): A 3D numpy array with shape (n, m, p), where each 2D array represents a data point.
        
        Returns:
        numpy.ndarray: A 1D numpy array containing the distances to the core for each 2D data point.
        """
        # Initialize an empty list to store distances to the core
        distance_to_core = []

        # Iterate over each 2D data point in the 3D array
        for i in range(data.shape[0]):
            # Select the current 2D data point
            current_data = data[i, :, :]

            # Find the indices of the maximum value in the current 2D data point
            ortho_idx, stream_idx = np.unravel_index(np.argmax(current_data, axis=None), current_data.shape)

            # Calculate the differences in x and y coordinates
            y_diff_mm = ortho_idx - 4 * 4.77
            x_diff_mm = stream_idx - 12 * 10

            # Calculate the Euclidean distance (norm) from the center to the maximum value position
            norm = np.sqrt(x_diff_mm ** 2 + y_diff_mm ** 2)

            # Append the calculated distance to the list
            distance_to_core.append(norm)

        # Convert the list of distances to a numpy array and return it
        return np.array(distance_to_core)

    def calculate_weighted_distances_to_core(self,data):
        """
        Calculate the weighted Euclidean distance from the center of each 2D array to each entry, 
        multiplied by the value of the entry.
        
        Parameters:
        data (numpy.ndarray): A 3D numpy array with shape (n, m, p), where each 2D array represents a data point.
        
        Returns:
        numpy.ndarray: A 1D numpy array containing the weighted distances to the core for each 2D data point.
        """
        # Initialize an empty list to store weighted distances to the core
        weighted_distance_to_core = []

        # Iterate over each 2D data point in the 3D array
        for i in range(data.shape[0]):
            # Select the current 2D data point
            current_data = data[i, :, :]

            # Calculate the center coordinates of the current 2D data point
            center_ortho = 4 * 4.77
            center_stream = 9 * 10

            # Calculate the differences in x and y coordinates for each entry
            y_diff_mm, x_diff_mm = np.mgrid[0:current_data.shape[0], 0:current_data.shape[1]]
            y_diff_mm = y_diff_mm.astype(float)
            x_diff_mm = x_diff_mm.astype(float)
            y_diff_mm -= center_ortho
            x_diff_mm -= center_stream

            # Calculate the Euclidean distance (norm) from the center to each entry
            distances = np.sqrt(x_diff_mm**2 + y_diff_mm**2)

            # Multiply the distances by the corresponding values in the current 2D data point
            weighted_distances = distances * current_data

            # Sum the weighted distances and append the result to the list
            weighted_distance_to_core.append(np.sum(weighted_distances))

        # Convert the list of weighted distances to a numpy array and return it
        return np.array(weighted_distance_to_core)

    def generate_boxplot_data(self,result, mode = 'all'):
        """
        Generates a boxplot DataFrame for the given result dictionary based on the specified mode.

        Args:
            result (dict): Dictionary containing 3D numpy arrays for each sex-genotype combination.
            mode (str): The mode to use when calculating distances to core. Options: 'all', 'max'.

        Returns:
            pd.DataFrame: Boxplot data as a DataFrame.

        Raises:
            ValueError: If the mode is not 'all' or 'max'.
        """
        boxplot_list = []

        for key, data in result.items():
            if mode == 'all':
                dist = self.calculate_weighted_distances_to_core(data)
            elif mode == 'max':
                dist = self.calculate_distances_to_core(data)
            else:
                raise ValueError(f'generate_boxplot_data: unknown mode: f{mode}')
            core_df = pd.DataFrame(dist, columns=['distance to center of stream, mm'])
            core_df['id'] = key[0] + key[1]
            core_df['sex'] = key[0]
            core_df['genotype'] = key[1]
            boxplot_list.append(core_df)

        return pd.concat(boxplot_list)
    #╔════════════════════════════════════════════════════════════════════════════════╗
    #║                             PLOTTING FUCNTIONS                                 ║
    #╚════════════════════════════════════════════════════════════════════════════════╝


    def plot_2d_histogram_with_marginals(self,x_axis, y_axis, data, interp_factor=2,clim=None):
        fig = plt.figure(figsize=(10, 7))
        gs = plt.GridSpec(6, 6, wspace=0.3, hspace=0.3)

        ax_main = fig.add_subplot(gs[1:-1, :-1])
        ax_main.set_aspect('equal')
        ax_top = fig.add_subplot(gs[0, :-1], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1:-1, -1], sharey=ax_main)

        # Interpolate data
        f = interp2d(x_axis, y_axis, data, kind='cubic')
        x_axis_new = np.linspace(x_axis.min(), x_axis.max(), x_axis.size * interp_factor)
        y_axis_new = np.linspace(y_axis.min(), y_axis.max(), y_axis.size * interp_factor)
        data_new = f(x_axis_new, y_axis_new)

        # Generate X and Y grids for pcolormesh
        x_grid, y_grid = np.meshgrid(np.append(x_axis_new, x_axis_new[-1] * 2 - x_axis_new[-2]), 
                                    np.append(y_axis_new, y_axis_new[-1] * 2 - y_axis_new[-2]))

        # Plot the interpolated 2D histogram on the main axis
        im = ax_main.pcolormesh(x_grid, y_grid, data_new, cmap='viridis',
                                vmin=clim[0] if clim else None,
                                vmax=clim[1] if clim else None)

        # Plot the marginal histograms
        ax_top.plot(x_axis, data.sum(axis=0), color='darkgray', alpha=0.7)
        ax_top.set_ylim(0,0.125)
        ax_right.plot(data.sum(axis=1), y_axis, color='darkgray', alpha=0.7)
        ax_right.set_xlim(0,0.275)

        # Remove ticks and labels from the marginal axes
        ax_top.set_xticks([])
        ax_right.set_yticks([])

        # Set axis labels
        ax_main.set_xlabel(f'X-axis (mm)')
        ax_main.set_ylabel(f'Y-axis (mm)')

        # Add a colorbar below the main axis
        divider = make_axes_locatable(ax_main)
        cax = divider.append_axes("bottom", size="5%", pad=0.5)
        cbar = fig.colorbar(im, cax=cax, label='Colorbar', orientation='horizontal')
        cbar.ax.xaxis.set_ticks_position('bottom')

        return fig

    def create_all_histograms(self,result, x_ax, y_ax,interp_factor =3,clim=(0.0, 0.03)):
        """
        Creates 2D histograms with marginals for the given result dictionary and returns a list of figures.

        Args:
            result (dict): Dictionary containing 3D numpy arrays for each sex-genotype combination.
            x_ax (np.array): X-axis values for the 2D histograms.
            y_ax (np.array): Y-axis values for the 2D histograms.
            interp_factor (int): Interpolation factor for 2D histogram plot.
            clim (tuple): Tuple containing the color limits for the 2D histogram plot.

        Returns:
            list: List of figures created for each sex-genotype combination.
        """
        fig_list = list()
        for key, data in result.items():
            com_data = np.sum(data, axis=0)
            com_data = com_data / com_data.sum()
            fig = self.plot_2d_histogram_with_marginals(x_ax, y_ax, com_data, interp_factor=interp_factor,clim=clim )
            fig.suptitle(key[0] + ' ' + key[1])
            fig_list.append(fig)
        return(fig_list)


    def plot_boxplot(self,box_df):
        """
        Plots a boxplot for the given DataFrame and returns the created figure.

        Args:
            box_df (pd.DataFrame): Input DataFrame containing boxplot data.

        Returns:
            matplotlib.figure.Figure: The created boxplot figure.
        """
        fig = plt.figure(figsize=(10, 10))
        sns.boxplot(data=box_df, x='genotype', y='distance to center of stream, mm', hue='sex', notch=False,
                    order=['rei-INT', 'rei-HT', 'rei-HM'], hue_order=['M', 'F'])

        return fig
    
    #╔════════════════════════════════════════════════════════════════════════════════╗
    #║                                    MAIN                                        ║
    #╚════════════════════════════════════════════════════════════════════════════════╝

    def save_result(self,boxplot_df, box_fig, hist_fig,fig_path,data_path):
        """
        Saves the boxplot figure, histogram figures, and boxplot data to the specified paths.
        
        Parameters
        ----------
        boxplot_df : pd.DataFrame
            The boxplot data DataFrame.
        box_fig : matplotlib.figure.Figure
            The boxplot figure.
        hist_fig : list
            A list of histogram figures.
        fig_path : str
            The path to save the figures.
        data_path : str
            The path to save the data.
        """
        box_fig.savefig(f'{fig_path}CC_distance2stream.svg')

        c = 1
        for fig in hist_fig:
            fig.savefig(f'{fig_path}CC_histogram_f{c}.svg')
            c +=1

        boxplot_df.to_csv(f'{data_path}counter_current.csv')




    def main(self,fig_path,data_path, distance_mode= 'all',interp_factor =3,clim=(0.0, 0.03)):
        """
        Generates and saves all plots and statistics using the specified parameters.
        
        Parameters
        ----------
        fig_path : str
            The path to save the figures.
        data_path : str
            The path to save the data.
        distance_mode : str, optional, default: 'all'
            The mode for generating boxplot data ('all', 'upstream', or 'downstream').
        interp_factor : int, optional, default:3
        The interpolation factor for creating histograms.
        clim : tuple, optional, default: (0.0, 0.03)
        The color limits for the histogram plots (min_value, max_value).
        """

        _,x_ax,y_ax  = self.read_prob_density_csv(self.df.path2_probDensity[0])
        prob_desinty = self.generate_grouped_data()
        boxplot_df   = self.generate_boxplot_data(prob_desinty,distance_mode)
        box_fig      = self.plot_boxplot(boxplot_df)
        hist_fig     = self.create_all_histograms(prob_desinty,x_ax,y_ax,interp_factor =3,clim=(0.0, 0.03))
        self.save_result(boxplot_df, box_fig, hist_fig,fig_path,data_path)

