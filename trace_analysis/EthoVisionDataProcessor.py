import pandas as pd
import numpy as np


class EthovisionDataProcessor:
    def __init__(self, subject_df, speed_threshold=0.02, fps=25, margins = (5,15,5,15)):
        self.subject_df = subject_df
        self.speed_threshold = speed_threshold
        self.fps = fps
        self.left_margin   = margins[0]
        self.right_margin  = margins[1]
        self.bottom_margin = margins[2]
        self.top_margin    = margins[3]

    def add_day_number(self):
        """
        Adds a new 'Day_number' column to the subject DataFrame based on the 'Start_time' column.
        The 'Day_number' column starts at 0 for the earliest date and increments by one for each
        unique date in the 'Start_time' column.
        """
        # Convert 'Start_time' column to datetime format
        self.subject_df['Start_time'] = pd.to_datetime(self.subject_df['Start_time'], format='%m/%d/%Y %H:%M:%S.%f')

        # Extract only the date from 'Start_time' column and drop duplicates
        unique_dates = self.subject_df['Start_time'].dt.normalize().drop_duplicates().sort_values()

        # Create a dictionary mapping unique dates to their day number
        day_number_mapping = {date: i for i, date in enumerate(unique_dates)}

        # Add a new 'Day_number' column based on the mapping
        self.subject_df['Day_number'] = self.subject_df['Start_time'].dt.normalize().map(day_number_mapping)


    def calculate_speed(self):
        """
        Calculates the speed in cm per second based on the first derivative of the columns
        'X_center_cm', 'Y_center_cm', and 'Recording_time_s', and adds the speed to the
        subject DataFrame as a new column called 'speed_cmPs'.
        """

        # Initialize a new column for the speed
        self.subject_df['speed_cmPs'] = np.nan

        # Iterate through the unique Day_numbers
        for day_number in self.subject_df['Day_number'].unique():
            # Extract the data for the current day
            day_data = self.subject_df[self.subject_df['Day_number'] == day_number]

            # Calculate the first derivative for the specified columns
            x_diff = day_data['X_center_cm'].diff()
            y_diff = day_data['Y_center_cm'].diff()
            time_diff = day_data['Recording_time_s'].diff()

            # Calculate the norm of the vector
            norm_vector = np.sqrt(x_diff ** 2 + y_diff ** 2)

            # Calculate the speed in cm per second
            speed_cmPs = norm_vector / time_diff

            # Insert a NaN value at the first row
            speed_cmPs.iloc[0] = np.nan

            # Update the 'speed_cmPs' column for the current day
            self.subject_df.loc[self.subject_df['Day_number'] == day_number, 'speed_cmPs'] = speed_cmPs

    def set_activity_status(self, speed_threshold=0.02):
        """
        Adds an 'activity' column to the subject DataFrame, indicating whether the
        subject's speed is greater than the given speed_threshold.

        Args:
            speed_threshold (float, optional): The threshold for considering a subject
                                               active. Defaults to 0.02.
        """

        self.subject_df['activity'] = self.subject_df.speed_cmPs > speed_threshold

    def compute_zones(self):
        """
        Calculates the boolean values for each zone (left, right, bottom, top) and
        adds them as new columns to the subject DataFrame.

        Args:
            left_margin (float): The boundary value for the left margin.
            right_margin (float): The boundary value for the right margin.
            bottom_margin (float): The boundary value for the bottom margin.
            top_margin (float): The boundary value for the top margin.
        """

        self.subject_df['in_left_margin']   = self.subject_df.X_center_cm < self.left_margin
        self.subject_df['in_right_margin']  = self.subject_df.X_center_cm > self.right_margin
        self.subject_df['in_bottom_margin'] = self.subject_df.Y_center_cm < self.bottom_margin        
        self.subject_df['in_top_margin']    = self.subject_df.Y_center_cm > self.top_margin
    

    def map_zones_to_integers(self):
        """
        Maps the boolean zone values in the input DataFrame to corresponding
        integers (similar to a numeric keypad layout) and adds a new column named
        'zone_map' to the DataFrame.
        """

        num_pad_direction = list()

        for i, row in self.subject_df.iterrows():
            if row['in_bottom_margin']:
                if row['in_left_margin']:
                    num_pad_direction.append(1)
                elif row['in_right_margin']:
                    num_pad_direction.append(3)
                else:
                    num_pad_direction.append(2)
            elif row['in_top_margin']:
                if row['in_left_margin']:
                    num_pad_direction.append(7)
                elif row['in_right_margin']:
                    num_pad_direction.append(9)
                else:
                    num_pad_direction.append(8)
            else:
                if row['in_left_margin']:
                    num_pad_direction.append(4)
                elif row['in_right_margin']:
                    num_pad_direction.append(6)
                else:
                    num_pad_direction.append(5)

        self.subject_df['zone_map'] = num_pad_direction

    def side_tigmotaxis(self):
        """
        Adds a new 'tigmo_taxis' column to the subject DataFrame, containing rows where activity is True
        and either in_left_margin or in_right_margin is True. This column represents a direct measure
        of tigmotaxis in the fish.
        """
        self.subject_df['tigmo_taxis'] = self.subject_df.loc[(self.subject_df['activity']) & (self.subject_df['in_left_margin'] | self.subject_df['in_right_margin'])]

    def true_freezing(self):
        """
        Adds a new 'freezing' column to the subject DataFrame, containing rows where in_bottom_margin is
        True and activity is False. This column quantifies "true" freezing, in which the fish sits on the
        bottom in rigor.
        """
        self.subject_df['freezing'] = self.subject_df.loc[(~self.subject_df['activity']) & (self.subject_df['in_bottom_margin'])]

    
    def latency_to_top(self,day_data):
        """
        Calculate the latency until the fish first enters the top zone.

        Returns:
            float: The index when the fish first enters the top zone.
        """
        first_top_entry = self.subject_df.loc[day_data['in_top_margin']].index[0]
        return first_top_entry/self.fps

    def side_zonening(self,day_data):
        """
        Calculate the total number of transitions in the zone map between the lateral zones.
        This represents the etho-vision version of the tigmotaxis analysis.

        Returns:
            int: The total number of transitions between the specified zones.
        """
        transitions = 0
        target_transitions = [(1, 4), (4, 1), (4, 7), (7, 4), (9, 6), (6, 9), (6, 3), (3, 6)]
        
        for i in range(len(day_data) - 1):
            current_zone = day_data['zone_map'].iloc[i]
            next_zone = day_data['zone_map'].iloc[i + 1]
            
            if (current_zone, next_zone) in target_transitions:
                transitions += 1
                
        return transitions


    def calculate_bout_metrics(self, data, column_name, total_time):
        """
        Calculate the median bout duration and the fraction of time spent in a given behavioral state
        (activity, freezing, or tigmotaxis) based on the specified boolean column.

        Args:
            data (pd.DataFrame): A DataFrame containing the data for a specific day.
            column_name (str): The name of the boolean column representing the behavioral state.
            total_time (float): The total recording time for the specific day.

        Returns:
            median_bout_duration (float): The median duration of bouts for the given behavioral state.
            fraction (float): The fraction of time spent in the given behavioral state.
        """
        bouts = data.groupby((data[column_name].shift() != data[column_name]).cumsum())
        bout_durations = bouts['Recording_time_s'].agg(np.ptp)
        boolean_value = bouts[column_name].first()

        median_bout_duration = bout_durations[boolean_value].median()
        fraction = bout_durations[boolean_value].sum() / total_time

        return median_bout_duration, fraction

    def process_data(self):
        """
        Process the subject data and compute various metrics for each day, such as median speed,
        median activity bout duration, activity fraction, median freezing bout duration, freezing fraction,
        median tigmotaxis bout duration, tigmotaxis fraction, tigmotaxis transitions, and latency to top.

        Returns:
            stats_df (pd.DataFrame): A DataFrame containing the computed metrics for each day.
        """
        self.add_day_number()
        self.calculate_speed()
        self.set_activity_status()
        self.compute_zones()
        self.map_zones_to_integers()
        self.side_tigmotaxis()
        self.true_freezing()

        median_speeds               = list()
        median_activity_durations   = list()
        activity_fractions          = list()
        median_freezing_durations   = list()
        freezing_fractions          = list()
        median_tigmotaxis_durations = list()
        tigmotaxis_fractions        = list()
        tigmotaxis_transitions      = list()
        time_to_top                 = list()

        for day in self.subject_df.Day_number.unique():
            day_data = self.subject_df.loc[self.subject_df.Day_number == day]
            total_time = day_data['Recording_time_s'].iloc[-1]

            # Median speed
            median_speed = day_data.loc[day_data['activity'], 'speed_cmPs'].median()
            median_speeds.append(median_speed)

            # Activity bout metrics
            median_activity_duration, activity_fraction = self.calculate_bout_metrics(day_data, 'activity', total_time)
            median_activity_durations.append(median_activity_duration)
            activity_fractions.append(activity_fraction)

            # Freezing bout metrics
            median_freezing_duration, freezing_fraction = self.calculate_bout_metrics(day_data, 'freezing', total_time)
            median_freezing_durations.append(median_freezing_duration)
            freezing_fractions.append(freezing_fraction)

            # Tigmo taxis bout metrics
            median_tigmotaxis_duration, tigmotaxis_fraction = self.calculate_bout_metrics(day_data, 'tigmo_taxis', total_time)
            median_tigmotaxis_durations.append(median_tigmotaxis_duration)
            tigmotaxis_fractions.append(tigmotaxis_fraction)

            # latency to go to top of the tank
            time_to_top.append(self.latency_to_top(day_data))

            # calculate tigmo_tactic_transitions
            tigmotaxis_transitions.append(self.side_zonening(day_data))

        stats_df = pd.DataFrame({
            'Day_number': self.subject_df.Day_number.unique(),
            'Median_speed_cmPs': median_speeds,
            'Median_activity_duration_s': median_activity_durations,
            'Activity_fraction': activity_fractions,
            'Median_freezing_duration_s': median_freezing_durations,
            'Freezing_fraction': freezing_fractions,
            'Median_tigmotaxis_duration_s': median_tigmotaxis_durations,
            'Tigmotaxis_fraction': tigmotaxis_fractions,
            'tigmotaxis_transitions': tigmotaxis_transitions,
            'latency_to_top_s': time_to_top
        })

        return stats_df
