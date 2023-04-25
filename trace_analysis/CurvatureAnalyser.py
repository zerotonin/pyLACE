import numpy as np
from scipy.signal import find_peaks

class CurvatureAnalyser:
    """
    A class used to analyze the curvature of a fish's midline based on a DataFrame containing its coordinates.
    
    Attributes
    ----------
    midline_df : pd.DataFrame
        The input DataFrame containing the fish's midline coordinates.
        
    Methods
    -------
    calculate_total_curvature(number_of_coordinates=10)
        Calculates the total curvature of the fish's midline for each row in the DataFrame.
        
    find_peak_amplitudes(curvature_vector, prominence_threshold)
        Finds the peak amplitudes in the curvature vector above the given prominence threshold.
        
    get_total_curvature_amps(prominence_threshold=0.5)
        Calculates the median, mean, and maximum peak amplitudes of the total curvature.
    """
    

    def __init__(self, midline_df):
        """
        Initializes the CurvatureAnalyser class with the input DataFrame.
        
        Parameters
        ----------
        midline_df : pd.DataFrame
            The input DataFrame containing the fish's midline coordinates.
        """
        self.midline_df = midline_df

    def calculate_total_curvature(self, number_of_coordinates=10):
        """
        Calculates the total curvature of the fish's midline for each row in the DataFrame.
        
        Parameters
        ----------
        number_of_coordinates : int, optional
            The number of coordinates in the midline (default is 10).
            
        Returns
        -------
        curvatures : np.array
            An array containing the total curvature for each row in the DataFrame.
        """
        curvatures = []
        for index, row in self.midline_df.iterrows():
            points = [(row[f'x_coord_{i}'], row[f'y_coord_{i}']) for i in range(number_of_coordinates)]  # Extract points
            tangents = np.diff(points, axis=0)  # Calculate tangents
            normalized_tangents = tangents / np.linalg.norm(tangents, axis=1, keepdims=True)  # Normalize tangents
            diff_normalized_tangents = np.diff(normalized_tangents, axis=0)  # Calculate differences between normalized tangents
            magnitudes = np.linalg.norm(diff_normalized_tangents, axis=1)  # Calculate magnitudes of differences
            total_curvature = np.sum(magnitudes)  # Sum the magnitudes to get the total curvature
            curvatures.append(total_curvature)
        return np.array(curvatures)


    def find_peak_amplitudes(self, curvature_vector, prominence_threshold):
        """
        Finds the peak amplitudes in the curvature vector above the given prominence threshold.

        Parameters
        ----------
        curvature_vector : np.array
            The curvature vector containing the curvature values.
        prominence_threshold : float
            The prominence threshold to consider a peak as significant.

        Returns
        -------
        peak_amplitudes : np.array
            The amplitudes of the detected peaks.
        """
        peaks, _ = find_peaks(curvature_vector, prominence=prominence_threshold)  # Find peaks in the curvature vector
        peak_amplitudes = curvature_vector[peaks]  # Get amplitudes of the peaks
        return peak_amplitudes


    def get_total_curvature_amps(self, prominence_threshold=0.5):
        """
        Calculates the median, mean, and maximum peak amplitudes of the total curvature.

        Parameters
        ----------
        prominence_threshold : float, optional
            The prominence threshold to consider a peak as significant (default is 0.5).

        Returns
        -------
        curvature_stats : dict
            A dictionary containing the median, mean, and maximum peak amplitudes of the total curvature.
        """
        total_curv = self.calculate_total_curvature()  # Calculate total curvature
        total_curv_amps = self.find_peak_amplitudes(total_curv, prominence_threshold)  # Find peak amplitudes
        median_curv_amp = np.nanmedian(total_curv_amps)  # Calculate the median amplitude
        mean_curv_amp = np.nanmean(total_curv_amps)  # Calculate the mean amplitude
        max_curv_amp = np.nanmax(total_curv_amps)  # Calculate the maximum amplitude
        max_curv_amp_index = np.nanargmax(total_curv_amps) # Find the index of the maximum amplitude



        return {'median_curv_amp': median_curv_amp, 'mean_curv_amp': mean_curv_amp, 
                'max_curv_amp': max_curv_amp,'max_curv_amp_index': max_curv_amp_index}