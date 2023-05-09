import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fish_data_base.fishDataBase as fishDataBase
import seaborn as sns
from data_handlers import matLabResultLoader
import plotting.fishPlot as fishPlot,cStartPlotter
import matplotlib.widgets as widgets
import glob
import data_handlers.mediaHandler as mediaHandler

#%%
db = fishDataBase.fishDataBase("/home/bgeurten/fishDataBase",'/home/bgeurten/fishDataBase/fishDataBase_cstart.csv')
#db.rebase_paths()
df = db.database

cs_plotter = cStartPlotter.cStartPlotter()


directory = "./"
svg_files = glob.glob(f"{directory}/*.svg")

good_trials = [75,164,261,326,345,378]

dfm = df.iloc[good_trials,:]
for i,row in dfm.iterrows():
    pass
mlr = matLabResultLoader.matLabResultLoader(row['path2_anaMat'])
raceInfo, traceContour, traceMidline, traceHead, traceTail, trace, bendability, binnedBend, saccs, trigAveSacc, medMaxVelocities =mlr.getData()

spike_df = pd.read_csv(row.path2_spike_train_df)

MHO = mediaHandler.mediaHandler(row.avi,'movie',row.fps,bufferSize = 2000)
frame = MHO.getFrame(0)
time_ax = fishPlot.makeTimeAxis(trace.shape[0],row.fps)

# Assuming spike_df is a pandas DataFrame with columns 'spike_peak_s' and 'instant_freq'
spike_peak_s = spike_df['spike_peak_s'].to_numpy()
instant_freq = spike_df['instant_freq'].to_numpy()
interp_instant_freq = np.interp(time_ax, spike_peak_s, instant_freq)

f,ax_list = cs_plotter.create_final_plot(spike_df, time_ax, trace, interp_instant_freq, traceContour, row.fps,frame)
ax_list[0].set_title(f'{row.genotype} {row.sex} ')
f.tight_layout()
plt.show()
#save_filename = f'./{i}_{row.genotype}_{row.sex}.svg'
#f.savefig(save_filename)
