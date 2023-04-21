
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

fN = r'/home/bgeurten/cstart_experiments/sufge1/Homozygous/male/movie9/HmM10 II.smr'
s2sr = spike2SimpleIO.spike2SimpleReader(fN)
s2sr.main()
segSav = spike2SimpleIO.segmentSaver(s2sr,'./testPanda.csv')
df = segSav.main()[0]
sd = SpikeDetector.SpikeDetector(df)
spike_train_df = sd.main()



df['camMon'] = df['camMon'].astype(int)
df['Pico Pum'] = df['Pico Pum'].astype(int)
df['Keyboard'] = df['Keyboard'].astype(int)


df['Signal stream 0'].plot()
#df['camMon'].plot()
#df['Pico Pum'].plot()
df['Keyboard'].plot()
df['Signal stream 0'].std()
plt.show()


