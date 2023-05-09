from data_base_analyser.EthoVisionExperimentSeries import EthoVisionExperimentSeries
import os
import pandas as pd
import numpy as np
import seaborn as sns
import glob
import matplotlib.pyplot as plt
import scipy.ndimage
from matplotlib.colors import LogNorm
import matplotlib.cm as cm

#[print(x.shape) for x in histograms]
# Usage
tag = 'habituation2023'
parent_directory = '/home/bgeurten/ethoVision_database/'

# Compile data daywise
etho_vision_analysis = EthoVisionExperimentSeries(tag, parent_directory)
etho_vision_analysis.process_and_save()

db_position = f'{parent_directory}{tag}_daywise_analysis.csv'
df = pd.read_csv(db_position)

histogram_file_positions = find_npy_files(parent_directory)

fishID, hists = load_normed_histograms(histogram_file_positions)


male_hists,female_hists = sort_hists_by_sex(hists,fishID,df)
male_hists = normalise_histograms(np.nanmedian(male_hists,axis=3))
female_hists = normalise_histograms(np.nanmedian(female_hists,axis=3))

create_daywise_histograms(male_hists)
create_daywise_histograms(female_hists)

for topic in ['Median_speed_cmPs', 'Gross_speed_cmPs',
       'Median_activity_duration_s', 'Activity_fraction',
       'Median_freezing_duration_s', 'Freezing_fraction',
       'Median_top_duration_s', 'Top_fraction', 'Median_bottom_duration_s',
       'Bottom_fraction', 'Median_tigmotaxis_duration_s',
       'Tigmotaxis_fraction', 'Tigmotaxis_transitions', 'Latency_to_top_s',
       'Distance_travelled_cm']:
    
    create_vertical_box_stripplot(df,'Day_number',topic,'Sex',('M','F'))
    plt.show()


create_vertical_box_stripplot(df,'Day_number','Top_fraction','Sex',('M','F'))
plt.show()



def create_top_fraction_lineplot(df,measure):
    """
    Creates a line plot for the Top_fraction over Day_number for each fish separately,
    with individual males in blue and females in red, based on the Tank_number, ID, and Sex columns in the DataFrame df.

    Args:
        df (pandas.DataFrame): The DataFrame to use for plotting.
    """
    # Group the data by Tank_number, ID, and Sex
    groups = df.groupby(["Tank_number", "ID", "Sex"])

    # Loop through the groups and plot the lines with the corresponding colors and markers
    for (tank_num, fish_id, sex), group in groups:
        color = "blue" if sex == "M" else "red"
        marker = "o" if group[measure].iloc[0] >= 0.4 else "x"
        label = f"{tank_num}, {fish_id}"

        # Create the plot
        fig, ax = plt.subplots()
        ax.plot(group["Day_number"], group[measure], color=color, label=label, marker=marker)

        # Add labels and title
        ax.set_xlabel("Day number")
        ax.set_ylabel(measure)
        ax.set_title(f"Fish ID: {fish_id}, Tank number: {tank_num}")
        ax.set_ylim((0,500))

        # Add legend and markers
        ax.legend()
        ax.axhline(y=0.4, color="gray", linestyle="--")
        ax.plot([], [], color=color, marker=marker, label="Fraction above 0.4")

        # Show the plot
        plt.show()

create_top_fraction_lineplot(df,'Tigmotaxis_transitions')