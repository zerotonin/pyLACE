
import sqlite3
import pandas as pd


def get_unique_combinations(db_connection):
    """
    Retrieves the unique combinations of values in the columns 'Tank_number', 'Sex', and 'Subject ID'
    from the SQLite database.

    Args:
        db_connection (sqlite3.Connection): A SQLite database connection.

    Returns:
        DataFrame: A DataFrame containing the unique combinations of 'Tank_number', 'Sex', and 'ID' values.
    """
    query = """
    SELECT DISTINCT Tank_number, Sex, "ID"
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

# Open an existing SQLite database
db_name = "/home/bgeurten/fishDataBase/alex_ethovision_data.db"
db_connection = open_database(db_name)

if db_connection:
    # Get the unique combinations of 'Tank_number', 'Sex', and 'Subject ID' values
    unique_combinations = get_unique_combinations(db_connection)
    print(unique_combinations)

    # Close the SQLite database connection
    db_connection.close()
else:
    print("Could not open the database")
