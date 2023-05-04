import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from fish_data_base.EthoVisionSQLdataBase import EthoVisionSQLdataBase
from trace_analysis.EthoVisionDataProcessor import EthovisionDataProcessor
from plotting.IndividualAnalysisReportEthoVision import IndividualAnalysisReportEthoVision
import pandas as pd
import numpy as np
import math



def save_numpy_array(n_array, filename):
    np.save(filename, n_array)

def save_dataframe(df,filename):
    df.to_csv(filename)

def save_figure(fig,filename):
    fig.savefig(filename, format='svg')


# Open an existing SQLite database
tag = 'habituation2023'
db_name = f"/home/bgeurten/ethoVision_database/{tag}_ethovision_data.db"


ev_db = EthoVisionSQLdataBase(db_name)
if ev_db.db_connection:
    # Get the unique combinations of 'Tank_number', 'Sex', and 'Subject ID' values
    unique_fish = ev_db.get_unique_subjects()
    print(unique_fish)
    for i,row in unique_fish.iterrows():
        pass
    subject_df = ev_db.get_data_for_subject(row.Tank_number,row.ID)
    print(subject_df)
    evp = EthovisionDataProcessor(subject_df)
    result_df,histograms = evp.process_data(tank_height=20.5,tank_width=20.5)
    subject_df = evp.subject_df
    print(result_df)

    reporter = IndividualAnalysisReportEthoVision(result_df,histograms)
    rep_figs = reporter.report()
    plt.show()
    # Close the SQLite database connection
    ev_db.close_connection()
else:
    print("Could not open the database")

