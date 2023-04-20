
import numpy as np
import pandas as pd
from pandas.core import api
from pandas.core.arrays.boolean import BooleanArray
import quantities as pq
import data_handlers.spike2SimpleIO as spike2SimpleIO 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

fN = '/home/bgeurten/cstart_trials_rei/Homozygous/Male/movie2/rei_cstHmM2_11-2019R.smr/HmM10 II.smr'
s2sr = spike2SimpleIO.spike2SimpleReader(fN)
s2sr.main()
segSav = spike2SimpleIO.segmentSaver(s2sr,'./testPanda.csv')
df = segSav.main()[0]



df['camMon'] = df['camMon'].astype(int)
df['Pico Pum'] = df['Pico Pum'].astype(int)
df['Keyboard'] = df['Keyboard'].astype(int)


df['Signal stream 0'].plot()
#df['camMon'].plot()
#df['Pico Pum'].plot()
df['Keyboard'].plot()
df['Signal stream 0'].std()
plt.show()


