import pandas as pd
from data_handlers.EthoVisionReader import EthoVisionReader
import sqlite3
import os
from pathlib import Path
from tqdm import tqdm

def get_all_xlsx_files(folder):
    """
    Searches for all .xlsx files in the given folder and its subdirectories.

    Args:
        folder (str): The path to the folder to search for .xlsx files.

    Returns:
        list: A list of .xlsx file paths found in the folder and its subdirectories.
    """
    xlsx_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".xlsx"):
                xlsx_files.append(os.path.join(root, file))
    return xlsx_files

def read_all_ethovision_files_to_pandas(xlsx_files):
    """
    Reads all EthoVision Excel files in the given list using the EthoVisionReader class.

    Args:
        xlsx_files (list): A list of .xlsx file paths to read.

    Returns:
        DataFrame: A DataFrame containing the concatenated data from all the Excel files.
    """
    all_data = []

    for file in tqdm(xlsx_files,desc='reading files'):
        etho_vision_reader = EthoVisionReader(file)
        file_data = etho_vision_reader.main()
        all_data.append(file_data)

    return pd.concat(all_data)


def read_all_ethovision_files_to_sql(xlsx_files, db_connection):
    """
    Reads all EthoVision Excel files in the given list using the EthoVisionReader class
    and stores the data in the provided SQLite database.

    Args:
        xlsx_files (list): A list of .xlsx file paths to read.
        db_connection (sqlite3.Connection): A SQLite database connection.

    Returns:
        None
    """
    for file in  tqdm(xlsx_files,desc='reading files'):
        etho_vision_reader = EthoVisionReader(file)
        file_data = etho_vision_reader.main()
        file_data.to_sql('ethovision_data', db_connection, if_exists='append', index=False)

def create_database(db_name):
    """
    Creates a new SQLite database with the specified name.

    Args:
        db_name (str): The name of the SQLite database file.

    Returns:
        sqlite3.Connection: A connection to the SQLite database.
    """
    return sqlite3.connect(db_name)

# Specify the folder to search for .xlsx files
folder = "/media/bgeurten/TOSHIBA_EXT/alex/raw_data/"

# Get a list of all .xlsx files in the folder and its subdirectories
xlsx_files = get_all_xlsx_files(folder)

# Create a SQLite database and connect to it
db_name = "/home/bgeurten/fishDataBase/alex_ethovision_data.db"
db_connection = create_database(db_name)

# Read all EthoVision Excel files and store the data in the SQLite database
read_all_ethovision_files_to_sql(xlsx_files, db_connection)

# Close the database connection
db_connection.close()

