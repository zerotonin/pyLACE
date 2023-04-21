import pandas as pd
from scipy.signal import find_peaks
import numpy as np

class SpikeDetector:
    """
    A class used to detect spikes in a single-channel electrophysiology signal.

    Attributes
    ----------
    df_signal : pd.DataFrame
        The input DataFrame containing the electrophysiology signal.
    spike_train_df : pd.DataFrame
        The DataFrame containing the detected spike times, amplitudes, and instantaneous frequencies.
    stimulus_occurence_s : float
        The time (in seconds) when the stimulus occurred.
    min_latency_s : float
        The minimum latency (in seconds) between the stimulus occurrence and a spike.

    Methods
    -------
    find_peaks_in_df(threshold)
        Finds the peaks in the signal with a given prominence threshold.
        
    get_peak_time_and_amp(noise_std_factor=1.5)
        Returns a DataFrame with peak times and amplitudes of positive and negative spikes.
        
    calculate_instantaneous_spike_freq(spike_df)
        Calculates the instantaneous spike frequency based on inter-spike intervals.
        
    separate_M_units()
        Separates the detected spikes into Mauthner and other categories based on their amplitudes.
    
    get_timing_from_keyboard()
        Retrieves the time stamp (in seconds) when the user activated the stimulus and triggered the experiment.

    main(noise_factor=1.5)
        Runs the spike detection process and returns the DataFrame containing the detected spikes and their properties.
    """

    def __init__(self, df_signal, min_latency_s=0.005):
        """
        Initializes the SpikeDetector class with the input DataFrame.

        Parameters
        ----------
        df_signal : pd.DataFrame
            The input DataFrame containing the electrophysiology signal.
        min_latency_s : float, optional
            The minimum latency (in seconds) between the stimulus occurrence and a spike (default is 0.005).
        """
        self.df_signal = df_signal
        self.min_latency_s = min_latency_s
        self.spike_train_df = None
        self.stimulus_occurence_s = None


    def find_peaks_in_df(self, threshold):
        """
        Finds the peaks in the signal with a given prominence threshold.

        Parameters
        ----------
        threshold : float
            The prominence threshold for detecting peaks.

        Returns
        -------
        spike_df : pd.DataFrame
            A DataFrame containing the spike times, start and stop times, and amplitudes.
        """
        peak_pos, peak_data = find_peaks(self.df_signal['Signal stream 0'], prominence=threshold)
        peak_pos_time = self.df_signal.index[peak_pos].to_numpy()
        peak_start_time = self.df_signal.index[peak_data['left_bases']].to_numpy()
        peak_end_time = self.df_signal.index[peak_data['right_bases']].to_numpy()
        peak_amplitude = peak_data['prominences']
        spike_df = pd.DataFrame(np.stack([peak_pos_time, peak_start_time, peak_end_time, peak_amplitude]).T,
                   columns=['spike_peak_s', 'spike_start_s', 'spike_stop_s', 'amplitude_muV'])
        return spike_df

    def get_peak_time_and_amp(self, noise_std_factor=1.5):
        """
        Returns a DataFrame with peak times and amplitudes of positive and negative spikes.

        Parameters
        ----------
        noise_std_factor : float, optional
            The noise standard deviation factor for setting the prominence threshold (default is 1.5).

        Returns
        -------
        spike_df : pd.DataFrame
            A DataFrame containing the peak times and amplitudes of positive and negative spikes.
        """
        threshold = self.df_signal['Signal stream 0'].std() * noise_std_factor

        spike_df_positive = self.find_peaks_in_df(threshold=threshold)

        self.df_signal['Signal stream 0'] = self.df_signal['Signal stream 0'] * -1
        spike_df_negative = self.find_peaks_in_df(threshold=threshold)
        self.df_signal['Signal stream 0'] = self.df_signal['Signal stream 0'] * -1
        spike_df_negative.amplitude_mV = spike_df_negative.amplitude_muV * -1

        spike_df = pd.concat([spike_df_positive, spike_df_negative])
        spike_df = spike_df.sort_values(by='spike_peak_s')
        spike_df = spike_df.reset_index(drop=True)
        self.spike_train_df = spike_df

    @staticmethod
    def calculate_instantaneous_spike_freq(spike_df):
        """
        Calculates the instantaneous spike frequency based on inter-spike intervals.

        Parameters
        ----------
        spike_df : pd.DataFrame
            A DataFrame containing the amplitudes of positive and negative spikes.
        Returns
        -------
        instantaneous_frequency : np.array
            An array containing the instantaneous spike frequencies.
        """
        # Calculate inter-spike intervals
        inter_spike_intervals = np.diff(spike_df.spike_peak_s.to_numpy())
        # Calculate instantaneous frequency
        instantaneous_frequency = 1 / inter_spike_intervals
        return instantaneous_frequency
    
    def separate_M_units(self):
        """
        Separates the detected spikes into Mauthner and other categories based on their amplitudes.

        In the experiments, the fish were stimulated with a startle stimulus (an air blast from a micro injection pump),
        which triggers a flight reaction controlled by two giant fibers in the zebrafish spinal chord, called Mauthner neurons
        or M-cells. These spikes are much larger and usually above 7.5 microVolts in amplitude. This function subdivides spikes
        into Mauthner and other categories.
        """

        self.spike_train_df['spike_category'] = 'Other'
        self.spike_train_df.loc[self.spike_train_df.amplitude_muV.abs() > 7.5,'spike_category'] = 'Mauthner'
    
    def get_timing_from_keyboard(self):
        """
        Retrieves the time stamp (in seconds) when the user activated the stimulus
        and triggered the experiment.

        Returns
        -------
        stimulus_time_stamps : pd.Index
            The time stamps (in seconds) corresponding to the stimulus activation.
        """
        self.stimulus_occurence_s = self.df_signal.index[self.df_signal.Keyboard]
       
    
    def calculate_latency(self):
        """
        Calculates the latency between the stimulus occurrence and the first Mauthner spike,
        and the latency between the stimulus occurrence and the first other spike.
        Spikes that occur before the minimal latencies are ignored.

        Returns
        -------
        mauthner_latency : float
            The latency between the stimulus occurrence and the first Mauthner spike.
        other_latency : float
            The latency between the stimulus occurrence and the first other spike.
        """
        mauthner_latency = self.df_signal.index[-1]
        other_latency = self.df_signal.index[-1]

        for stimulus_time in self.stimulus_occurence_s:
            mauthner_candidates = self.spike_train_df[(self.spike_train_df.spike_category == "Mauthner") &
                                                    (self.spike_train_df.spike_peak_s >= stimulus_time + self.min_latency_s)]
            other_candidates = self.spike_train_df[(self.spike_train_df.spike_category == "Other") &
                                                    (self.spike_train_df.spike_peak_s >= stimulus_time + self.min_latency_s)]

            if not mauthner_candidates.empty:
                first_mauthner_spike = mauthner_candidates.iloc[0]
                mauthner_latency = first_mauthner_spike.spike_peak_s - stimulus_time

            if not other_candidates.empty:
                first_other_spike = other_candidates.iloc[0]
                other_latency = first_other_spike.spike_peak_s - stimulus_time

        return mauthner_latency, other_latency

    
    def main(self, noise_factor=1.5):
        """
        Runs the spike detection process and returns the DataFrame containing the detected spikes and their properties.

        Parameters
        ----------
        noise_factor : float, optional
            The noise standard deviation factor for setting the prominence threshold (default is 1.5).

        Returns
        -------
        spike_train_df : pd.DataFrame
            A DataFrame containing the detected spike times, amplitudes, and instantaneous frequencies.
        """
        # Detect spikes and their properties
        self.get_peak_time_and_amp(noise_factor)
        # Separate spikes into Mauthner and other categories
        self.separate_M_units()
        # Calculate instantaneous spike frequencies
        instant_freq = self.calculate_instantaneous_spike_freq(self.spike_train_df)
        instant_freq = np.insert(instant_freq, 0, 0)
        # Add instantaneous frequencies to the spike_train_df DataFrame
        self.spike_train_df['instant_freq'] = instant_freq
        # Get Stimulus occurence
        self.get_timing_from_keyboard()
        # Calculate latendy to spikes
        mauthner_latency, other_latency = self.calculate_latency()

        return self.spike_train_df
    
