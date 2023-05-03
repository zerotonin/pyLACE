import pandas as pd
import sqlite3


class EthoVisionSQLdataBase:
    def __init__(self, db_name):
        self.db_name = db_name
        self.db_connection = self.open_database()
        
    def open_database(self):
        """
        Opens an existing SQLite database with the specified name.

        Returns:
            sqlite3.Connection: A connection to the SQLite database.
        """
        try:
            db_connection = sqlite3.connect(self.db_name)
            return db_connection
        except sqlite3.Error as e:
            print(f"Error opening the database: {e}")
            return None

    def get_unique_subjects(self):
        """
        Retrieves the unique combinations of values in the columns 'Tank_number' and 'Subject ID'
        from the SQLite database.

        Returns:
            DataFrame: A DataFrame containing the unique combinations of 'Tank_number', and 'ID' values.
        """
        query = """
        SELECT DISTINCT Tank_number, "ID"
        FROM ethovision_data;
        """

        unique_combinations = pd.read_sql_query(query, self.db_connection)

        return unique_combinations

    def get_data_for_subject(self, tank_number, id_val):
        """
        Retrieves all data for a specific combination of 'Tank_number' and 'ID'
        from the SQLite database.

        Args:
            tank_number (str): The 'Tank_number' value.
            id_val (str): The 'ID' value.

        Returns:
            DataFrame: A DataFrame containing all data for the specified combination.
        """
        query = """
        SELECT *
        FROM ethovision_data
        WHERE Tank_number = ?  AND "ID" = ?;
        """

        data_for_combination = pd.read_sql_query(query, self.db_connection, params=(tank_number, id_val))
        data_for_combination.X_center_cm = pd.to_numeric(data_for_combination.X_center_cm, errors='coerce')
        data_for_combination.Y_center_cm = pd.to_numeric(data_for_combination.Y_center_cm, errors='coerce')

        return self.sort_dataframe(data_for_combination)

    def sort_dataframe(self, df):
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
