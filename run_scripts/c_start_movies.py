import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fish_data_base.fishDataBase as fishDataBase
import seaborn as sns
from data_handlers import matLabResultLoader
import plotting.fishPlot as fishPlot,cStartPlotter
import matplotlib.widgets as widgets
import glob,os
import data_handlers.mediaHandler as mediaHandler
from tqdm import tqdm

#%%
db = fishDataBase.fishDataBase("/home/bgeurten/fishDataBase",'/home/bgeurten/fishDataBase/fishDataBase_cstart.csv')
#db.rebase_paths()
df = db.database

cs_plotter = cStartPlotter.cStartPlotter()


directory = "./"
svg_files = glob.glob(f"{directory}/*.svg")

good_trials = [75,164,261,326,345,378]
offsets = [(0,0),(0,0),(0,0),(-10,0),(-10,0),(-10,0)]
trace_video_disperity = [0,0,0,0,-523,0]
round_robin_correction =[0,0,0,-3715,-4027,-3866]
dfm = df.iloc[good_trials,:]
#dfm = df[df['genotype'].str.contains('sufge1')]
c = 0
#%%
for i,row in tqdm(dfm.iterrows(), total=dfm.shape[0],desc='Still making movies...'):

    mlr = matLabResultLoader.matLabResultLoader(row['path2_anaMat'])
    traceInfo, traceContour, traceMidline, traceHead, traceTail, trace, bendability, binnedBend, saccs, trigAveSacc, medMaxVelocities =mlr.getData()
    
    # correct trace and video disperaty
    trace = np.roll(trace,trace_video_disperity[c],axis=0)
    traceContour = np.roll(traceContour,trace_video_disperity[c],axis=0)

    # correct round robin role
    trace = np.roll(trace,round_robin_correction[c],axis=0)
    traceContour = np.roll(traceContour,round_robin_correction[c],axis=0)

    spike_df = pd.read_csv(row.path2_spike_train_df)

    time_ax = fishPlot.makeTimeAxis(trace.shape[0],row.fps)

    # Assuming spike_df is a pandas DataFrame with columns 'spike_peak_s' and 'instant_freq'
    spike_peak_s = spike_df['spike_peak_s'].to_numpy()
    instant_freq = spike_df['instant_freq'].to_numpy()
    interp_instant_freq = np.interp(time_ax, spike_peak_s, instant_freq)

    # Make animation file name# Get the base file name
    filename = os.path.basename(row.avi).replace(' ','_')
    animation_filepath = os.path.join("/home/bgeurten/", f'{row.genotype}_{filename}')
    animation = cs_plotter.create_animated_plot(spike_df, time_ax, trace, interp_instant_freq, traceContour, 
                                                row.fps,row.avi,animation_filepath,contour_offSet=offsets[c],
                                                round_robin_offset=round_robin_correction[c])
    c+=1


#%%
import data_handlers.spike2SimpleIO as spike2SimpleIO
import trace_analysis.SpikeDetector as SpikeDetector
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


cs_plotter = cStartPlotter.cStartPlotter()

for i,row in tqdm(dfm.iterrows(), total=dfm.shape[0],desc='Plot Raw Ephys'):

    # Read the Spike2 file
    s2sr = spike2SimpleIO.spike2SimpleReader(row.path2_smr)
    s2sr.main()
    # Save segments to a CSV file
    segSav = spike2SimpleIO.segmentSaver(s2sr, 'no csv file will be produced')
    df = segSav.main()[0]
    # Detect spikes
    sd = SpikeDetector.SpikeDetector(df)
    spike_train_df, spike_properties = sd.main()

    spike_df = pd.read_csv(row.path2_spike_train_df)
    # Assuming spike_df is a pandas DataFrame with columns 'spike_peak_s' and 'instant_freq'
    spike_peak_s = spike_df['spike_peak_s'].to_numpy()
    instant_freq = spike_df['instant_freq'].to_numpy()


    # Create a figure object
    fig = plt.figure(figsize=(8,5))
    # Create a gridspec object
    gs = gridspec.GridSpec(10, 1, figure=fig)  # 10 rows, 1 column
    # Create subplots
    ax1 = fig.add_subplot(gs[:9, :])  # Top subplot taking 9 parts out of 10
    ax2 = fig.add_subplot(gs[9:, :])  # Bottom subplot taking 1 part out of 10

    sd.df_signal.plot(ax=ax1,legend=False)
    ax1.set_xlabel("")
    ax1.set_xlim((0,5))
    ax1.set_ylim((-2,2))
    ax1.set_xticklabels([])
    ax1.plot([0,5],[sd.threshold,sd.threshold],'k--')
    ax1.plot([0,5],[sd.threshold*-1,sd.threshold*-1],'k--')
    ax1.set_ylabel('field potential, muV ')  # remove x-axis label
    cs_plotter.plot_spike_occurrences(spike_df, ax2)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_ylabel('')  # remove x-axis label
    plt.tight_layout()
    #plt.show()


    filename = os.path.basename(row.avi).replace(' ','_').split('.')[0]
    figure_filepath = os.path.join("/home/bgeurten/", f'{row.genotype}_{filename}.svg')
    fig.savefig(figure_filepath)

# %%
