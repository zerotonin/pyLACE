
import sqlite3
import pandas as pd


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

    return data_for_combination


def add_day_number(df):
    """
    Adds a new 'Day_number' column to the input DataFrame based on the 'Start_time' column.
    The 'Day_number' column starts at 0 and increments by one for each new day in the 'Start_time' column.

    Args:
        df (DataFrame): A DataFrame with a 'Start_time' column containing date-time values.

    Returns:
        DataFrame: The input DataFrame with an additional 'Day_number' column.
    """
    # Convert 'Start_time' column to datetime format
    df['Start_time'] = pd.to_datetime(df['Start_time'], format='%m/%d/%Y %H:%M:%S.%f')

    # Calculate the minimum date in the 'Start_time' column
    min_date = df['Start_time'].min().normalize()

    # Add a new 'Day_number' column based on the difference between 'Start_time' and the minimum date
    df['Day_number'] = (df['Start_time'].dt.normalize() - min_date).dt.days

    return df



# Open an existing SQLite database
db_name = "/home/bgeurten/fishDataBase/alex_ethovision_data.db"
db_connection = open_database(db_name)

if db_connection:
    # Get the unique combinations of 'Tank_number', 'Sex', and 'Subject ID' values
    unique_fish = get_unique_subjects(db_connection)
    print(unique_fish)
    tank_numbers,id_val = unique_fish.iloc[1,:]
    subject_df = get_data_for_subject(db_connection,tank_numbers,id_val )
    subject_df = add_day_number(subject_df)
    print(subject_df)

    # Close the SQLite database connection
    db_connection.close()
else:
    print("Could not open the database")
