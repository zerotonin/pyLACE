
import numpy as np
import pandas as pd
from pandas.core import api
from pandas.core.arrays.boolean import BooleanArray
import quantities as pq
import data_handlers.spike2SimpleIO as spike2SimpleIO 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
def find_peaks_in_df(df_signal,threshold):
    peak_pos, peak_data = find_peaks(df_signal['Signal stream 0'], prominence=threshold)
    peak_pos_time = df_signal.index[peak_pos].to_numpy()
    peak_start_time = df_signal.index[peak_data['left_bases']].to_numpy()
    peak_end_time = df_signal.index[peak_data['right_bases']].to_numpy()
    peak_amplitude= peak_data['prominences']
    spike_df = pd.DataFrame(np.stack([peak_pos_time, peak_start_time, peak_end_time, peak_amplitude]).T,
               columns= ['spike_peak_s','spike_start_s','spike_stop_s','amplitude_mV'])
    return spike_df

def get_peak_time_and_amp(df_signal, noise_std_factor = 1.5):
    threshold= df_signal['Signal stream 0'].std()* noise_std_factor

    spike_df_positive = find_peaks_in_df(df_signal,threshold=threshold)

    df_signal['Signal stream 0'] = df_signal['Signal stream 0']*-1
    spike_df_negative = find_peaks_in_df(df_signal,threshold=threshold)
    df_signal['Signal stream 0'] = df_signal['Signal stream 0']*-1
    spike_df_negative.amplitude_mV = spike_df_negative.amplitude_mV*-1

    spike_df = pd.concat([spike_df_positive,spike_df_negative])
    spike_df = spike_df.sort_values(by='spike_peak_s')
    spike_df = spike_df.reset_index(drop=True)
    return spike_df

def calculate_instantenous_spike_freq(spike_df):
    # Calculate inter-pulse intervals
    inter_spike_intervals = np.diff(spike_df.spike_peak_s.to_numpy())
    # Calculate instantaneous frequency
    instantaneous_frequency = 1 / inter_spike_intervals
    return instantaneous_frequency

fN = '/home/bgeurten/cstart_trials_rei/Homozygous/Male/movie2/rei_cstHmM2_11-2019R.smr'
s2sr = spike2SimpleIO.spike2SimpleReader(fN)
s2sr.main()
segSav = spike2SimpleIO.segmentSaver(s2sr,'./testPanda.csv')
df = segSav.main()[0]
spike_df = get_peak_time_and_amp(df)

df['camMon'] = df['camMon'].astype(int)
df['Pico Pum'] = df['Pico Pum'].astype(int)
df['Keyboard'] = df['Keyboard'].astype(int)


df['Signal stream 0'].plot()
#df['camMon'].plot()
#df['Pico Pum'].plot()
df['Keyboard'].plot()
df['Signal stream 0'].std()
plt.show()


