
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_unique_subjects(db_connection):
    """
    Retrieves the unique combinations of values in the columns 'Tank_number' and 'Subject ID'
    from the SQLite database.

    Args:
        db_connection (sqlite3.Connection): A SQLite database connection.

    Returns:
        DataFrame: A DataFrame containing the unique combinations of 'Tank_number', and 'ID' values.
    """
    query = """
    SELECT DISTINCT Tank_number, "ID"
    FROM ethovision_data;
    """

    unique_combinations = pd.read_sql_query(query, db_connection)

    return unique_combinations

def open_database(db_name):
    """
    Opens an existing SQLite database with the specified name.

    Args:
        db_name (str): The name of the SQLite database file.

    Returns:
        sqlite3.Connection: A connection to the SQLite database.
    """
    try:
        db_connection = sqlite3.connect(db_name)
        return db_connection
    except sqlite3.Error as e:
        print(f"Error opening the database: {e}")
        return None

def get_data_for_subject(db_connection, tank_number, id_val):
    """
    Retrieves all data for a specific combination of 'Tank_number' and 'ID'
    from the SQLite database.

    Args:
        db_connection (sqlite3.Connection): A SQLite database connection.
        tank_number (str): The 'Tank_number' value.
        subject_id (str): The 'ID' value.

    Returns:
        DataFrame: A DataFrame containing all data for the specified combination.
    """
    query = """
    SELECT *
    FROM ethovision_data
    WHERE Tank_number = ?  AND "ID" = ?;
    """

    data_for_combination = pd.read_sql_query(query, db_connection, params=(tank_number, id_val))
    data_for_combination.X_center_cm = pd.to_numeric(data_for_combination.X_center_cm, errors='coerce')
    data_for_combination.Y_center_cm = pd.to_numeric(data_for_combination.Y_center_cm, errors='coerce')

    return data_for_combination

def sort_dataframe(df):
    """
    Sorts the input DataFrame first by the 'Start_time' column and then by the 'Recording_time' column.

    Args:
        df (DataFrame): A DataFrame with 'Start_time' and 'Recording_time' columns.

    Returns:
        DataFrame: The sorted DataFrame.
    """
    # Convert 'Start_time' and 'Recording_time' columns to datetime format, if not already in that format
    df['Start_time'] = pd.to_datetime(df['Start_time'], errors='ignore')

    # Sort DataFrame by 'Start_time' and 'Recording_time' columns
    sorted_df = df.sort_values(by=['Start_time', 'Recording_time_s'])

    return sorted_df

def add_day_number(df):
    """
    Adds a new 'Day_number' column to the input DataFrame based on the 'Start_time' column.
    The 'Day_number' column starts at 0 for the earliest date and increments by one for each
    unique date in the 'Start_time' column.

    Args:
        df (DataFrame): A DataFrame with a 'Start_time' column containing date-time values.

    Returns:
        DataFrame: The input DataFrame with an additional 'Day_number' column.
    """
    # Convert 'Start_time' column to datetime format
    df['Start_time'] = pd.to_datetime(df['Start_time'], format='%m/%d/%Y %H:%M:%S.%f')

    # Extract only the date from 'Start_time' column and drop duplicates
    unique_dates = df['Start_time'].dt.normalize().drop_duplicates().sort_values()

    # Create a dictionary mapping unique dates to their day number
    day_number_mapping = {date: i for i, date in enumerate(unique_dates)}

    # Add a new 'Day_number' column based on the mapping
    df['Day_number'] = df['Start_time'].dt.normalize().map(day_number_mapping)

    return df


def calculate_speed(df):
    """
    Calculates the speed in cm per second based on the first derivative of the columns
    'X_center_cm', 'Y_center_cm', and 'Recording_time_s', and adds the speed to the
    input DataFrame as a new column called 'speed_cmPs'.

    Args:
        df (DataFrame): A DataFrame with columns 'X_center_cm', 'Y_center_cm', and 'Recording_time_s'.

    Returns:
        DataFrame: A DataFrame with an additional 'speed_cmPs' column containing the calculated speed.
    """
    # Calculate the first derivative for the specified columns
    x_diff = df['X_center_cm'].diff()
    y_diff = df['Y_center_cm'].diff()
    time_diff = df['Recording_time_s'].diff()

    # Calculate the norm of the vector
    norm_vector = np.sqrt(x_diff**2 + y_diff**2)

    # Calculate the speed in cm per second
    speed_cmPs = norm_vector / time_diff

    # Insert a NaN value at the first row
    speed_cmPs.iloc[0] = np.nan

    # Add the 'speed_cmPs' column to the DataFrame
    df['speed_cmPs'] = speed_cmPs

    return df

def set_activity_status(subject_df, speed_threshold=0.02):
    """
    Adds an 'activity' column to the input DataFrame, indicating whether the
    subject's speed is greater than the given speed_threshold.

    Args:
        subject_df (pd.DataFrame): A DataFrame containing subject data.
        speed_threshold (float, optional): The threshold for considering a subject
                                           active. Defaults to 0.02.
    """

    subject_df['activity'] = subject_df.speed_cmPs > speed_threshold

def compute_zones(subject_df, left_margin, right_margin, bottom_margin, top_margin):
    """
    Calculates the boolean values for each zone (left, right, bottom, top) and
    adds them as new columns to the input DataFrame.

    Args:
        subject_df (pd.DataFrame): A DataFrame containing subject data.
        left_margin (float): The boundary value for the left margin.
        right_margin (float): The boundary value for the right margin.
        bottom_margin (float): The boundary value for the bottom margin.
        top_margin (float): The boundary value for the top margin.
    
    Returns:
        pd.DataFrame: The input DataFrame with added columns for each zone.
    """

    subject_df['in_left_margin'] = subject_df.X_center_cm < left_margin
    subject_df['in_right_margin'] = subject_df.X_center_cm > right_margin
    subject_df['in_bottom_margin'] = subject_df.Y_center_cm < bottom_margin
    subject_df['in_top_margin'] = subject_df.Y_center_cm > top_margin
    return subject_df

def map_zones_to_integers(subject_df):
    """
    Maps the boolean zone values in the input DataFrame to corresponding
    integers (similar to a numeric keypad layout) and adds a new column named
    'zone_map' to the DataFrame.

    Args:
        subject_df (pd.DataFrame): A DataFrame containing subject data with
                                   boolean columns for each zone.
    
    Returns:
        pd.DataFrame: The input DataFrame with an added 'zone_map' column
                      containing integer values representing zones.
    """

    num_pad_direction = list()

    for i, row in subject_df.iterrows():
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

    subject_df['zone_map'] = num_pad_direction
    return subject_df


def latency_to_top(subject_df, fps):
    """
    Calculate the latency until the fish first enters the top zone.

    Args:
        subject_df (pd.DataFrame): A DataFrame containing 'in_top_margin' column.
        fps (int): Frames per second of the recorded data.

    Returns:
        float: The latency in seconds until the fish first enters the top zone.
    """
    first_top_entry = subject_df.loc[subject_df['in_top_margin']].index[0]
    latency = first_top_entry / fps
    return latency

def side_tigmotaxis(subject_df):
    """
    Calculate the total count of rows where activity is True and either in_left_margin or in_right_margin is True.
    Thereby it is a direct major of tigmotaxis in the fish.

    Args:
        subject_df (pd.DataFrame): A DataFrame containing 'activity', 'in_left_margin', and 'in_right_margin' columns.

    Returns:
        int: The total count of rows where activity and (in_left_margin or in_right_margin) are True.
    """
    active_side_rows = subject_df.loc[(subject_df['activity']) & (subject_df['in_left_margin'] | subject_df['in_right_margin'])]
    return len(active_side_rows)

def side_zonening(subject_df):
    """
    Calculate the total number of transitions in the zone map between the lateral zones.
    This represents the etho-vision version of the tigmotaxis analysis.

    Args:
        subject_df (pd.DataFrame): A DataFrame containing 'zone_map' column.

    Returns:
        int: The total number of transitions between the specified zones.
    """
    transitions = 0
    target_transitions = [(1, 4), (4, 1), (4, 7), (7, 4), (9, 6), (6, 9), (6, 3), (3, 6)]
    
    for i in range(len(subject_df) - 1):
        current_zone = subject_df['zone_map'].iloc[i]
        next_zone = subject_df['zone_map'].iloc[i + 1]
        
        if (current_zone, next_zone) in target_transitions:
            transitions += 1
            
    return transitions

def true_freezing(subject_df):
    """
    Calculate the total count of rows where in_bottom_margin is True and activity is False.
    Thereby we quantify "true" freezing in which the fish sits on the bottom in rigor.

    Args:
        subject_df (pd.DataFrame): A DataFrame containing 'activity' and 'in_bottom_margin' columns.

    Returns:
        int: The total count of rows where in_bottom_margin is True and activity is False.
    """
    freezing_rows = subject_df.loc[(~subject_df['activity']) & (subject_df['in_bottom_margin'])]
    return len(freezing_rows)


def get_speed_and_duration_stats(subject_df, speed_threshold):
    """
    Calculate the median speed above the given threshold, median bout duration above
    and below the threshold, and the total duration below the threshold for each unique day
    in the given DataFrame and return the results as a single-row DataFrame.

    Args:
        subject_df (pd.DataFrame): A DataFrame containing 'Day_number', 'Recording_time_s', 'X_center_cm', and 'Y_center_cm' columns.
        speed_threshold (float): The speed threshold to filter the data.

    Returns:
        pd.DataFrame: A single-row DataFrame with the calculated values as columns.
    """
    # Calculate speed and add it to the DataFrame
    subject_df = calculate_speed(subject_df)

    # Set activity status based on speed_threshold
    set_activity_status(subject_df, speed_threshold)

    median_speeds              = list()
    median_bout_duration_above = list()
    median_bout_duration_below = list()
    total_duration_below       = list()
    trial_list                 = list()

    for i in subject_df.Day_number.unique():
        day_data = subject_df.loc[subject_df.Day_number == i]

        # Calculate median speed above threshold
        speed_above_threshold = day_data.loc[day_data['activity'], 'speed_cmPs']
        median_speeds.append(np.nanmedian(speed_above_threshold))

        # Calculate bout durations above and below threshold
        bouts = day_data.groupby((day_data['activity'].shift() != day_data['activity']).cumsum())
        bout_durations = bouts['Recording_time_s'].agg(np.ptp)

        # Calculate median bout duration above and below threshold
        median_bout_duration_above.append(np.nanmedian(bout_durations[bouts['activity'].first()]))
        median_bout_duration_below.append(np.nanmedian(bout_durations[~bouts['activity'].first()]))

        # Calculate total duration below threshold
        total_duration_below.append(bout_durations[~bouts['activity'].first()].sum())

        # trial numbers
        trial_list.append(i)

    # Create a dictionary with the calculated values
    result_dict = {
        'trial': trial_list,
        'median_speeds': median_speeds,
        'median_bout_duration_above': median_bout_duration_above,
        'median_bout_duration_below': median_bout_duration_below,
        'total_duration_below': total_duration_below
    }

    # Convert the dictionary to a single-row DataFrame
    result_df = pd.DataFrame(result_dict)

    return result_df


# Open an existing SQLite database
db_name = "/home/bgeurten/fishDataBase/alex_ethovision_data.db"
db_connection = open_database(db_name)

if db_connection:
    # Get the unique combinations of 'Tank_number', 'Sex', and 'Subject ID' values
    unique_fish = get_unique_subjects(db_connection)
    print(unique_fish)
    tank_numbers,id_val = unique_fish.iloc[50,:]
    subject_df = get_data_for_subject(db_connection,tank_numbers,id_val )
    subject_df = add_day_number(subject_df)
    subject_df = sort_dataframe(subject_df)
    subject_df = set_activity_threshold(subject_df)
    result = get_speed_and_duration_stats(subject_df,25)
    print(subject_df)

    # Close the SQLite database connection
    db_connection.close()
else:
    print("Could not open the database")


subject_df.speed_cmPs.hist(bins=np.linspace(0,5,50)) 
plt.show()

speed_meds = list()
for i in subject_df.Day_number.unique():
    speed_meds.append(np.nanmedian(subject_df.loc[subject_df.Day_number == i,'speed_cmPs']))

plt.plot(speed_meds)
plt.show()