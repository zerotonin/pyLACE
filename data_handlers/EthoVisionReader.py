import pandas as pd
import numpy as np
from scipy.interpolate import griddata

class EthoVisionReader:
    """
    A class to read EthoVision Excel files and extract trajectory and metadata.

    Attributes:
        filename (str): The filename of the Excel file to read.
        excel_data (dict): A dictionary containing the data from the Excel file.

    Author: B. Geurten
    Date: 28th April 2023
    """


    def __init__(self, filename, tank_height = 20.5, tank_width = 20.5, correction_mode = False, correction_factor = 1/38.9683801):
        """
        Constructs the EthoVisionReader object with the given filename.

        Args:
            filename (str): The filename of the Excel file to read.
            tank_width (float): The width of the tank.
            tank_height (float): The height of the tank.
        """
        self.filename = filename
        self.excel_data = self.read_file()
        
        self.correction_mode = correction_mode
        self.correction_factor = correction_factor

        self.tank_height = tank_height
        self.tank_width = tank_width
        self.set_tank_corner_coordinates()

    def set_tank_corner_coordinates(self):
        """
        Sets the corner coordinates for each tank to be used for converting
        the EthoVision coordinate system to a tank coordinate system.

        The tank coordinate system has its origin in the lower left corner,
        and the Y-axis pointing upwards.

        The corner coordinates are stored as a list of dictionaries in the
        attribute `self.tank_coordinates`. Each dictionary represents a tank
        and contains the following keys: 'lower left', 'upper left', 'upper right', 
        and 'lower right'. The values are tuples with the X and Y coordinates.

        For example:
        self.tank_coordinates = [
            {'lower left': (-43.35, -22.5), 'upper left': (-43.28, -2.14), ...},
            {'lower left': (-20.57, -22.9), 'upper left': (-20.78, -6.02), ...},
            ...
        ]

        This function is called in the __init__ method of the EthoVisionReader class.

        Returns:
            None
        """
        tank_0 = {'lower left':(-43.35,-22.5),'upper left':(-43.28,-2.14),'upper right':(-20.91,-5.94),'lower right':(-20.98,-22.02)}
        tank_1 = {'lower left':(-20.57,-22.09),'upper left':(-20.78,-6.02),'upper right':(0.21,-6.02),'lower right':(0.21,-22.02)}
        tank_2 = {'lower left':(0.76,-22.02),'upper left':(0.69,-6.14),'upper right':(21.47,-6.14),'lower right':(-21.56,-21.95)}
        tank_3 = {'lower left':(22.02,-29.95),'upper left':(22.02,-6.14),'upper right':(44.11,-6.28),'lower right':(43.83,-22.98)}
        self.tank_coordinates = [tank_0,tank_1,tank_2,tank_3]

    def read_file(self):
        """
        Reads the Excel file and stores the data in a dictionary.

        Returns:
            dict: A dictionary containing the data from the Excel file.
        """
        return pd.read_excel(self.filename, sheet_name=None)

    def get_trajectory(self, sheet_data):
        """
        Extracts the trajectory data from the given sheet data.

        Args:
            sheet_data (DataFrame): The data from a single sheet in the Excel file.

        Returns:
            DataFrame: A DataFrame containing the trajectory data.
        """

        # Find the index of the 'Trial time' in the 1st column
        index = (sheet_data.iloc[:,0] == 'Trial time').idxmax()
        column_heads = sheet_data.iloc[index,:].to_list()
        column_units = sheet_data.iloc[index+1,:].to_list()
        col_combi = list(zip(column_heads, column_units))
        column_names = [f'{x[0]}_{x[1]}'.replace(' ','_') for x in col_combi]

        return pd.DataFrame(sheet_data.iloc[index+2::,:].to_numpy(), columns=column_names)

    def get_meta_data(self, sheet_data, df):
        """
        Adds metadata to the given DataFrame.

        Args:
            sheet_data (DataFrame): The data from a single sheet in the Excel file.
            df (DataFrame): A DataFrame containing the trajectory data.

        Returns:
            DataFrame: A DataFrame containing the trajectory data with metadata.
        """
        meta_keys = ['Tank_number', 'Sex', 'ID', 'Start time', 'Arena ID', 'Trial ID', 'Subject ID']
        
        for key in meta_keys:
            value = sheet_data.loc[sheet_data.iloc[:, 0] == key].iloc[:, 1].values
            df[key.replace(' ', '_')] = value[0] if len(value) > 0 else None

        return df
   
    def apply_correction_factor(self, df):
        """
        Applies the correction factor to the 'X_center_cm' and 'Y_center_cm' columns
        in the given DataFrame if the correction mode is set to true.

        Args:
            df (pd.DataFrame): A DataFrame containing 'X_center_cm' and 'Y_center_cm' columns.
        
        Returns:
            pd.DataFrame: A DataFrame with the corrected 'X_center_cm' and 'Y_center_cm' columns.
        """
        if self.correction_mode:
            df['X_center_cm']    = df['X_center_cm'].astype(float) * self.correction_factor
            df['Y_center_cm']    = df['Y_center_cm'].astype(float) * self.correction_factor
            df['Area_cm²']       = df['Area_cm²'].astype(float) * self.correction_factor
            df['Areachange_cm²'] = df['Areachange_cm²'].astype(float) * self.correction_factor
            df['Distance_moved_cm'] = df['Distance_moved_cm'].replace('-', 0).astype(float) * self.correction_factor

        return df

    
    def interpolate_coordinates(self, meta_data):
        """
        Interpolates the 'X_center_cm' and 'Y_center_cm' coordinates in df_trajectory,
        using the given tank_coordinates, to new coordinates based on a tank with the
        specified tank_width and tank_height.

        Args:
            meta_data (pd.DataFrame): A DataFrame containing the metadata.
        Returns:
            pd.DataFrame: A DataFrame with the interpolated 'X_center_cm' and 'Y_center_cm' coordinates.
        """

        # Get the arena number from the metadata
        arena_ID = int(meta_data['Arena_ID'].iloc[0])

        # Get the tank corners for the corresponding tank_number
        corners = self.tank_coordinates[arena_ID]

        # Define the tank corners for the new coordinate system
        new_corners = np.array([(0, 0), (0, self.tank_height), ( self.tank_width,  self.tank_height), ( self.tank_width, 0)])

        # Prepare the points for interpolation
        source_points = np.array([corners['lower left'], corners['upper left'], corners['upper right'], corners['lower right']])
        target_points = np.array(meta_data[['X_center_cm', 'Y_center_cm']])

        # Perform the interpolation
        interpolated_points = griddata(source_points, new_corners, target_points)

        # Update the 'X_center_cm' and 'Y_center_cm' columns with the interpolated coordinates
        meta_data['X_center_cm'] = interpolated_points[:, 0]
        meta_data['Y_center_cm'] = interpolated_points[:, 1]

        return meta_data

    def main(self):
        """
        Main function to process the data from the Excel file.

        Returns:
            DataFrame: A DataFrame containing the concatenated data from all sheets.
        """
        df_list = list()

        for sheet_name, sheet_data in self.excel_data.items():
            if sheet_data.iloc[-1, 0] != 'No samples logged for this track!':
                # get trajectory data
                df_trajectory = self.get_trajectory(sheet_data)
                # Apply the correction factor
                df_trajectory = self.apply_correction_factor(df_trajectory) 
                # combine trajectory with important meta data
                df_meta_data = self.get_meta_data(sheet_data, df_trajectory)
                #interpolate to tank coordinates
                df_meta_data = self.interpolate_coordinates(df_meta_data)
                #save
                df_list.append(df_meta_data)

        final_data = pd.concat(df_list)
        return final_data

# Example usage:
# filename = "/home/bgeurten/Downloads/Raw_data-2023_setup-Trial1.xlsx"
# etho_vision_reader = EthoVisionReader(filename,correction_mode=True)
# final_data = etho_vision_reader.main()
# print(final_data)