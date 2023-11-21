import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fish_data_base.fishDataBase as fishDataBase
import seaborn as sns
from data_handlers import matLabResultLoader
import plotting.fishPlot as fishPlot,plotting.cStartPlotter as cStartPlotter
import matplotlib.widgets as widgets
import glob

def on_button_save(event, fig, save_filename):
    fig.savefig(save_filename)
    plt.close(fig)

def on_button_close(event, fig):
    plt.close(fig)
#%%
db = fishDataBase.fishDataBase("/home/bgeurten/fishDataBase",'/home/bgeurten/fishDataBase/fishDataBase_cstart.csv')
#db.rebase_paths()
df = db.database

cs_plotter = cStartPlotter.cStartPlotter()
#%%
for i,row in df.iterrows():
    if i > 0:
        try:
            mlr = matLabResultLoader.matLabResultLoader(row['path2_anaMat'])
            traceInfo, traceContour, traceMidline, traceHead, traceTail, trace, bendability, binnedBend, saccs, trigAveSacc, medMaxVelocities =mlr.getData()
        
            spike_df = pd.read_csv(row.path2_spike_train_df)

            time_ax = fishPlot.makeTimeAxis(trace.shape[0],row.fps)
            # Assuming spike_df is a pandas DataFrame with columns 'spike_peak_s' and 'instant_freq'
            spike_peak_s = spike_df['spike_peak_s'].to_numpy()
            instant_freq = spike_df['instant_freq'].to_numpy()
            interp_instant_freq = np.interp(time_ax, spike_peak_s, instant_freq)

            f,ax_list = cs_plotter.create_final_plot(spike_df, time_ax, trace, interp_instant_freq, traceContour, row.fps)
            ax_list[0].set_title(f'{row.genotype} {row.sex} ')
            f.tight_layout()
        
            # Create buttons
            ax_close = plt.axes([0.35, 0.05, 0.1, 0.075])
            ax_save_close = plt.axes([0.55, 0.05, 0.1, 0.075])
            button_close = widgets.Button(ax_close, 'Close')
            button_save_close = widgets.Button(ax_save_close, 'Save and Close')

            # Connect button click events to the event handler
            save_filename = f'./{i}_{row.genotype}_{row.sex}.svg'
            button_close.on_clicked(lambda event: on_button_close(event, f))
            button_save_close.on_clicked(lambda event: on_button_save(event, f, save_filename))

            plt.show()
            print(i)
        except:
            pass

#%%


directory = "./"
svg_files = glob.glob(f"{directory}/*.svg")

good_trials = [int(x.split('_')[0].split('/')[1]) for x in svg_files]

for i,row in df.iterrows():
    if i  in good_trials:
        mlr = matLabResultLoader.matLabResultLoader(row['path2_anaMat'])
        traceInfo, traceContour, traceMidline, traceHead, traceTail, trace, bendability, binnedBend, saccs, trigAveSacc, medMaxVelocities =mlr.getData()
    
        spike_df = pd.read_csv(row.path2_spike_train_df)

        time_ax = fishPlot.makeTimeAxis(trace.shape[0],row.fps)
        # Assuming spike_df is a pandas DataFrame with columns 'spike_peak_s' and 'instant_freq'
        spike_peak_s = spike_df['spike_peak_s'].to_numpy()
        instant_freq = spike_df['instant_freq'].to_numpy()
        interp_instant_freq = np.interp(time_ax, spike_peak_s, instant_freq)

        f,ax_list = cs_plotter.create_final_plot(spike_df, time_ax, trace, interp_instant_freq, traceContour, row.fps)
        ax_list[0].set_title(f'{row.genotype} {row.sex} ')
        f.tight_layout()
        save_filename = f'./{i}_{row.genotype}_{row.sex}.svg'
        f.savefig(save_filename)
        



sns.set_theme(style="ticks")
 

for category in ['latency_to_m_cell', 'latency_to_others', 'm_cell_spikes',
       'median_spike_instFreq_Hz', 'other_spikes']:
    

    # Initialize the figure with a logarithmic x axis
    f, ax = plt.subplots(figsize=(7, 6))

    # Plot the orbital period with horizontal boxes
    sns.boxplot(x="genotype", y=category, data=df, hue='sex',
                whis=[0, 100], width=.6, palette="vlag",notch=True)

    # Add in points to show each observation
    sns.stripplot(x="genotype", y=category, data=df, hue='sex',
                size=4, color=".3", linewidth=0)

    # Tweak the visual presentation
    ax.xaxis.grid(True)
    ax.set(ylabel=category)
    sns.despine(trim=True, left=True)

plt.show()