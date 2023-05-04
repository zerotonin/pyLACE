from fish_data_base.EthoVisionSQLdataBase import EthoVisionSQLdataBase
from trace_analysis.EthoVisionDataProcessor import EthovisionDataProcessor
from plotting.IndividualAnalysisReportEthoVision import IndividualAnalysisReportEthoVision
import os
import numpy as np




def save_numpy_array(n_array, filename):
    np.save(filename, n_array)

def save_dataframe(df,filename):
    df.to_csv(filename, index = False)

def save_figure(fig,filename):
    fig.savefig(filename, format='svg')

def make_subject_directory_string(parent_directory,tank_number,fish_id):
    formatted_tank_number = "{:02}".format(int(tank_number))
    new_directory = f'{parent_directory}tankNum_{formatted_tank_number}__fishID_{fish_id}/'

    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    return new_directory

def save_report_figures(fig_handles,subject_dir_str):
    names = ['bout_metrics_01','velocities_distance','zones','spatial_histogram']
    for i in range(4):
        save_figure(fig_handles[i],f'{subject_dir_str}{names[i]}.svg')

# Open an existing SQLite database
tag = 'habituation2023'
parent_directory = '/home/bgeurten/ethoVision_database/'
db_name = f"{parent_directory}{tag}_ethovision_data.db"


ev_db = EthoVisionSQLdataBase(db_name)
if ev_db.db_connection:
    # Get the unique combinations of 'Tank_number', 'Sex', and 'Subject ID' values
    unique_fish = ev_db.get_unique_subjects()

    for i,row in unique_fish.iterrows():
        pass
    
    # Get subject data
    subject_df = ev_db.get_data_for_subject(row.Tank_number,row.ID)
    
    # process data
    evp = EthovisionDataProcessor(subject_df)
    result_df,histograms = evp.process_data(tank_height=20.5,tank_width=20.5)
    subject_df = evp.subject_df

    # produce figures
    reporter = IndividualAnalysisReportEthoVision(result_df,histograms)
    rep_figs = reporter.report()

    # save output
    subject_directory = make_subject_directory_string(parent_directory,row.Tank_number,row.ID)
    save_report_figures(rep_figs,subject_directory)
    save_numpy_array(histograms,f'{subject_directory}spatial_histograms.npy')
    save_dataframe(result_df,f'{subject_directory}collated_data.csv')
    save_dataframe(subject_df,f'{subject_directory}trajectory_data.csv')



    # Close the SQLite database connection
    ev_db.close_connection()
else:
    print("Could not open the database")

