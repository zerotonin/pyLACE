from data_base_analyser.EthoVisionExperimentSeries import EthoVisionExperimentSeries
from plotting.DaywiseAnalysis import DaywiseAnalysis
from plotting.FishHabituationProfiler import FishHabituationProfiler
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
#etho_vision_analysis = EthoVisionExperimentSeries(tag, parent_directory)
#etho_vision_analysis.process_and_save()

db_position = f'{parent_directory}{tag}_daywise_analysis.csv'
df = pd.read_csv(db_position)

dwa= DaywiseAnalysis(df,parent_directory)
#dwa.create_spatial_histograms()
#dwa.create_box_strip_plots()
#plt.show()

fhp = FishHabituationProfiler(df)
figs, fishIDs = fhp.check_habituation()

figure_directory = '/home/bgeurten/ethoVision_database/habituation_figs/'

for i in range(len(figs)):
    figs[i].savefig(f'{figure_directory}/{fishIDs[i]}.svg')


