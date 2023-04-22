
import numpy as np
import pandas as pd
from pandas.core import api
from pandas.core.arrays.boolean import BooleanArray
import quantities as pq
import data_handlers.spike2SimpleIO as spike2SimpleIO 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import trace_analysis.SpikeDetector as SpikeDetector
import fish_data_base.fishDataBase  as fishDataBase

"""
#fN = r'/home/bgeurten/cstart_experiments/sufge1/Homozygous/male/movie9/HmM10 II.smr'
fN = r'/home/bgeurten/cstart_experiments/sufge1/IntWild/male/movie5/IntM6.smr'
s2sr = spike2SimpleIO.spike2SimpleReader(fN)
s2sr.main()
segSav = spike2SimpleIO.segmentSaver(s2sr,'./testPanda.csv')
df = segSav.main()[0]
sd = SpikeDetector.SpikeDetector(df)
spike_train_df = sd.main()
"""

#%%
db = fishDataBase.fishDataBase("/home/bgeurten/fishDataBase",'/home/bgeurten/fishDataBase/fishDataBase_cstart.csv')
#db.rebase_paths()


multiFileFolder = '/home/bgeurten/cstart_experiments/sufge1/'
# Experiment types CCur couynter current , Ta tapped, Unt untapped, cst, c-startz
db.run_multi_trace_folder(multiFileFolder,'sufge1','cst','08-2019',start_at=0,gui_correction=False)


