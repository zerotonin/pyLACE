import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from fish_data_base.EthoVisionSQLdataBase import EthoVisionSQLdataBase
from trace_analysis.EthoVisionDataProcessor import EthovisionDataProcessor
from plotting.IndividualAnalysisReportEthoVision import IndividualAnalysisReportEthoVision
from tqdm import tqdm


class EthoVisionExperimentSeries:
    """
    A class for processing and saving data from an EthoVision experiment series.

    Attributes:
        tag (str): The tag for the experiment series.
        parent_directory (str): The directory containing the experiment data.
        db_name (str): The name of the SQLite database file.
        ev_db (EthoVisionSQLdataBase): An instance of the EthoVision SQL database.
    """
    def __init__(self, tag, parent_directory):
        """
        Initializes the EthoVisionExperimentSeries class with a tag and parent directory.
        
        Args:
            tag (str): The tag for the experiment series.
            parent_directory (str): The directory containing the experiment data.
        """
        self.tag = tag
        self.parent_directory = parent_directory
        self.db_name = os.path.join(self.parent_directory, f"{self.tag}_ethovision_data.db")
        self.ev_db = EthoVisionSQLdataBase(self.db_name)

    def save_numpy_array(self, n_array, filename):
        """
        Saves a NumPy array to a file.
        
        Args:
            n_array (numpy.ndarray): The NumPy array to save.
            filename (str): The file path where the array will be saved.
        """
        np.save(filename, n_array)

    def save_dataframe(self, df, filename):
        """
        Saves a pandas DataFrame to a CSV file.
        
        Args:
            df (pandas.DataFrame): The DataFrame to save.
            filename (str): The file path where the DataFrame will be saved.
        """
        df.to_csv(filename, index=False)

    def save_figure(self, fig, filename):
        """
        Saves a matplotlib figure to an SVG file.
        
        Args:
            fig (matplotlib.figure.Figure): The figure to save.
            filename (str): The file path where the figure will be saved.
        """

        fig.savefig(filename, format='svg')

    def make_subject_directory_string(self, tank_number, fish_id):
        """
        Creates a subject directory string based on tank number and fish ID.
        
        Args:
            tank_number (int): The tank number.
            fish_id (str): The fish ID.
        
        Returns:
            str: The subject directory string.
        """
        formatted_tank_number = "{:02}".format(int(tank_number))
        new_directory = os.path.join(self.parent_directory, f'tankNum_{formatted_tank_number}__fishID_{fish_id}')

        if not os.path.exists(new_directory):
            os.makedirs(new_directory)

        return new_directory

    def save_report_figures(self, fig_handles, subject_dir_str):
        """
        Saves the report figures to disk.
        
        Args:
            fig_handles (list): A list of figure handles for the report figures.
            subject_dir_str (str): The subject directory string for saving the figures.
        """
        names = ['bout_metrics_01', 'velocities_distance', 'zones', 'spatial_histogram']
        for i in range(4):
            self.save_figure(fig_handles[i], os.path.join(subject_dir_str, f'{names[i]}.svg'))

    def process_and_save(self):
        """
        Processes and saves the data and figures for the EthoVision experiment series.
        """
        if self.ev_db.db_connection:
            # Get the unique combinations of 'Tank_number', 'Sex', and 'Subject ID' values
            unique_fish = self.ev_db.get_unique_subjects()
            result_list = list()
            with tqdm(total=unique_fish.shape[0], desc='individual analysis') as pbar:
                for i, row in unique_fish.iterrows():
                    pbar.set_description(f"Tank: {row.Tank_number}, ID: {row.ID}")
                    
                    # Get subject data
                    subject_df = self.ev_db.get_data_for_subject(row.Tank_number, row.ID)

                    # process data
                    evp = EthovisionDataProcessor(subject_df)
                    result_df, histograms = evp.process_data(tank_height=20.5, tank_width=20.5)
                    subject_df = evp.subject_df

                    # produce figures
                    reporter = IndividualAnalysisReportEthoVision(result_df, histograms)
                    rep_figs = reporter.report()

                    # save output
                    subject_directory = self.make_subject_directory_string(row.Tank_number, row.ID)
                    self.save_report_figures(rep_figs, subject_directory)
                    self.save_numpy_array(histograms, os.path.join(subject_directory, 'spatial_histograms.npy'))
                    self.save_dataframe(result_df, os.path.join(subject_directory, 'collated_data.csv'))
                    self.save_dataframe(subject_df, os.path.join(subject_directory, 'trajectory_data.csv'))
                    result_list.append(result_df)

                    # Close all figures
                    plt.close('all')
                    
                    pbar.update(1)            
                


            
            #Save out results
            result_list = pd.concat(result_list)
            self.save_dataframe(result_list, os.path.join(self.parent_directory, f'{self.tag}_daywise_analysis.csv'))
            
            # Close the SQLite database connection
            self.ev_db.close_connection()

        else:
            print("Could not open the database")
