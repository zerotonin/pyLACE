import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from fish_data_base.EthoVisionSQLdataBase import EthoVisionSQLdataBase
from trace_analysis.EthoVisionDataProcessor import EthovisionDataProcessor
import pandas as pd
import numpy as np
import math

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

import matplotlib.pyplot as plt

def plot_bout_metrics(result_df):
    # Extract the necessary data from the result_df
    days = result_df['Day_number']
    metrics = [
        ('Median_activity_duration_s', 'Activity_fraction', 'Activity'),
        ('Median_freezing_duration_s', 'Freezing_fraction', 'Freezing'),
        ('Median_tigmotaxis_duration_s', 'Tigmotaxis_fraction', 'Tigmotaxis')
    ]

    # Create a figure with subplots for bout metrics and fractions
    fig, axes = plt.subplots(len(metrics), 2, figsize=(12, 12))
    fig.tight_layout(pad=4)

    for i, (duration_col, fraction_col, title_prefix) in enumerate(metrics):
        # Plot median durations
        axes[i, 0].plot(days, result_df[duration_col], marker='o')
        axes[i, 0].set_xlabel('Day number')
        axes[i, 0].set_ylabel(f'Median {title_prefix.lower()} duration (s)')
        axes[i, 0].set_title(f'Median {title_prefix} Duration over Days')

        # Plot fractions
        axes[i, 1].plot(days, result_df[fraction_col], marker='o')
        axes[i, 1].set_xlabel('Day number')
        axes[i, 1].set_ylabel(f'{title_prefix.lower()} fraction')
        axes[i, 1].set_title(f'{title_prefix} Fraction over Days')

    return fig



def plot_velocity_metrics(result_df):
    # Extract the necessary data from the result_df
    days = result_df['Day_number']
    metrics = [
        ('Median_speed_cmPs', 'Gross_speed_cmPs', 'Velocity (cm/s)'),
        ('Distance_travelled_cm', '', 'Distance travelled (cm)'),
    ]

    # Create a figure with subplots for velocity metrics
    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 12))
    fig.tight_layout(pad=4)

    for i, (metric_col1, metric_col2, ylabel) in enumerate(metrics):
        if metric_col2 != '':
            # Plot metric data with two series in the same axis
            axes[i].plot(days, result_df[metric_col1], marker='o', label='Median')
            axes[i].plot(days, result_df[metric_col2], marker='o', label='Gross')
            axes[i].legend()
        else:
            # Plot metric data with a single series
            axes[i].plot(days, result_df[metric_col1], marker='o')
        
        axes[i].set_xlabel('Day number')
        axes[i].set_ylabel(ylabel)
        axes[i].set_title(f'{ylabel} over Days')

    return fig
def plot_bout_and_transition_metrics(result_df):
    # Extract the necessary data from the result_df
    days = result_df['Day_number']
    metrics = [
        ('Median_top_duration_s', 'Top duration (s)'),
        ('Top_fraction', 'Top fraction'),
        ('Median_bottom_duration_s', 'Bottom duration (s)'),
        ('Bottom_fraction', 'Bottom fraction'),
        ('Latency_to_top_s', 'Latency to top (s)'),
        ('Tigmotaxis_transitions', 'Tigmotaxis transitions')
    ]

    # Create a figure with subplots for bout and transition metrics
    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 18))
    fig.tight_layout(pad=4)

    for i, (metric_col, ylabel) in enumerate(metrics):
        # Plot metric data
        axes[i].plot(days, result_df[metric_col], marker='o')
        axes[i].set_xlabel('Day number')
        axes[i].set_ylabel(ylabel)

    return fig

def plot_normalized_histograms(histograms, tank_width, tank_height):
    num_days = histograms.shape[0]
    x_bins = np.linspace(0, tank_width, histograms.shape[1] + 1)
    y_bins = np.linspace(0, tank_height, histograms.shape[2] + 1)

    num_rows = int(math.ceil(math.sqrt(num_days)))
    num_cols = int(math.ceil(num_days / num_rows))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 6 * num_rows), sharex=True, sharey=True)
    fig.tight_layout(pad=2)

    # Normalize the histograms
    normalized_histograms = histograms / histograms.sum(axis=(1, 2), keepdims=True)

    # Flatten axes array for easy indexing
    axes_flat = axes.flatten()

    for i, ax in enumerate(axes_flat[:num_days]):
        im = ax.imshow(
            normalized_histograms[i].T,
            origin='lower',
            extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
            aspect='equal',
            cmap='plasma',
            norm=LogNorm(vmin=1e-5, vmax=1),  # Logarithmic normalization
        )
        ax.set_xlabel('Tank width (cm)')
        ax.set_title(f'Day {i + 1}')

    axes_flat[0].set_ylabel('Tank height (cm)')

    # Remove extra subplots if any
    for ax in axes_flat[num_days:]:
        ax.remove()

    # Add a colorbar
    cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Normalized frequency')

    return fig



# Usage: plot_bout_metrics(result_df)
f1 = plot_bout_metrics(result_df)
f2 = plot_velocity_metrics(result_df)
f3 = plot_bout_and_transition_metrics(result_df)
f4 = plot_normalized_histograms(histograms,20.5,20.5)
plt.show()