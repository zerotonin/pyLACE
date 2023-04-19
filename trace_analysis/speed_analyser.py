import pandas as pd
import numpy as np
import index_tools

class SpeedAnalyser:
    """
    A class to analyse the speed of fish from a given trace DataFrame.
    """

    def __init__(self, fps, dataframe=None):
        """
        Initializes the SpeedAnalyser class.

        Args:
            fps (float): Frames per second.
            dataframe (pd.DataFrame, optional): Input DataFrame with fish trace data. Defaults to None.
        """
        self.fps = fps
        self.trace_df = dataframe if dataframe is not None else pd.DataFrame()
        self.all_speed = pd.DataFrame()
        self.speed_analysis_df = pd.DataFrame()
        self.activity = pd.Series()
        self.cruise_speed = pd.DataFrame()

    def extract_fish_speeds(self):
        """Extracts fish speeds (thrust, slip, yaw) from the input trace DataFrame."""
        self.all_speed = self.trace_df[['thrust_m/s', 'slip_m/s', 'yaw_deg/s']]

    def set_activity_array(self, activity_threshold=(0.025, 0.025, 100)):
        """
        Sets the activity array based on the given activity threshold.

        Args:
            activity_threshold (tuple, optional): A tuple containing the thresholds for thrust, slip, and yaw. Defaults to (0.025, 0.025, 100).
        """
        self.activity = pd.DataFrame([
            self.all_speed['thrust_m/s'].abs() > activity_threshold[0],
            self.all_speed['slip_m/s'].abs() > activity_threshold[1],
            self.all_speed['yaw_deg/s'].abs() > activity_threshold[2]
        ]).transpose().any(axis='columns')

    def extract_cruise_speed(self):
        """Extracts cruise speeds based on the activity array."""
        self.cruise_speed = self.all_speed[self.activity]

    def calculate_torque(self, mode='cruise'):
        """
        Calculates the torque based on the given mode.

        Args:
            mode (str, optional): Mode for calculating torque. 'cruise' or 'all'. Defaults to 'cruise'.

        Returns:
            float: The calculated torque.
        """
        if mode == 'cruise':
            torque_data = self.cruise_speed
        elif mode == 'all':
            torque_data = self.all_speed
        else:
            raise ValueError(f'calculate_torque: unknown mode: {mode}')

        torque = np.median((torque_data['thrust_m/s'].abs() +
                            torque_data['slip_m/s'].abs()) /
                            torque_data['yaw_deg/s'].abs())

        return torque

    def calculate_central_speed_values(self, speed_df):
        """
        Calculates mean and median for given speed DataFrame.

        Args:
            speed_df (pd.DataFrame): DataFrame containing speed data.

        Returns:
            list: List containing mean and median values for thrust, slip, and yaw.
        """
        data = speed_df.abs().mean().tolist()
        data += speed_df.abs().median().tolist()
        return data

    def analyse_fish_speed(self):
        """
        Analyses fish speed from the input trace DataFrame and calculates meta speed values.

        Returns:
            dict: A dictionary containing the calculated values.
        """
        self.extract_fish_speeds()
        self.set_activity_array()
        self.extract_cruise_speed()

        data = self.calculate_central_speed_values(self.all_speed)
        data += self.calculate_central_speed_values(self.cruise_speed)

        activity_start_end = index_tools.bool_Seq2start_end_indices(self.activity)
        data.append(self.activity.sum() / self.fps)
        data.append(self.activity.sum() / self.activity.shape[0])
        data.append(self.activity[::-1].idxmax() / self.fps)
        data.append(self.calculate_torque())

        keys = ['thrust_mean_m/s', 'slip_mean_m/s', 'yaw_mean_m/s', 'thrust_median_m/s', 'slip_median_m/s', 'yaw_median_m/s', 'cruising_thrust_mean_m/s', 'cruising_slip_mean_m/s',
                'cruising_yaw_mean_m/s', 'cruising_thrust_median_m/s', 'cruising_slip_median_m/s', 'cruising_yaw_median_m/s', 'activity_duration_s', 'activity_fraction', 'sec_to_first_stop', 'torque']
        return dict(zip(keys, data))
