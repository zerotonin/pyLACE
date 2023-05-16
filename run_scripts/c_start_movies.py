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
#dfm = df.iloc[good_trials,:]
dfm = df[df['genotype'].str.contains('sufge1')]
c = 0
for i,row in tqdm(dfm.iterrows(), total=dfm.shape[0],desc='Still making movies...'):
    mlr = matLabResultLoader.matLabResultLoader(row['path2_anaMat'])
    raceInfo, traceContour, traceMidline, traceHead, traceTail, trace, bendability, binnedBend, saccs, trigAveSacc, medMaxVelocities =mlr.getData()

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
                                                row.fps,row.avi,animation_filepath,contour_offSet=offsets[c])
    c+=1
