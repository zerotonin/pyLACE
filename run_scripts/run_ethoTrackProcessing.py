import matplotlib.pyplot as plt
from fish_data_base.EthoVisionSQLdataBase import EthoVisionSQLdataBase
from trace_analysis.EthoVisionDataProcessor import EthovisionDataProcessor
import pandas as pd

# Open an existing SQLite database
db_name = "/home/bgeurten/fishDataBase/alex_ethovision_data.db"

ev_db = EthoVisionSQLdataBase(db_name)



if ev_db.db_connection:
    # Get the unique combinations of 'Tank_number', 'Sex', and 'Subject ID' values
    unique_fish = ev_db.get_unique_subjects()
    print(unique_fish)
    tank_numbers,id_val = unique_fish.iloc[50,:]
    subject_df = ev_db.get_data_for_subject(tank_numbers,id_val )
    print(subject_df)
    ev_dp = EthovisionDataProcessor(subject_df)
    result_df,histograms = ev_dp.process_data(tank_height=20.5,tank_width=20.5)
    print(result_df)
    # Close the SQLite database connection
    ev_db.close_connection()
else:
    print("Could not open the database")


result_df.columns