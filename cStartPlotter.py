import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from scipy.interpolate import make_interp_spline
from matplotlib import patches

class cStartPlotter:
    """
    A class for plotting the contours, spike occurrences, and two parameters in a single figure.
    Used for Garag et al 2023 A and B
    """
    def create_vertical_axes(self):
        """
        Create a figure with three vertically arranged axes (ax1, ax2, ax3), and a colorbar axis (cax1)
        next to ax1.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure object to draw on.
        ax1, cax1, ax2, ax3 : tuple of matplotlib.axes.Axes
            The Axes objects for the created subplots.
        """
        # Create the figure
        fig = plt.figure()

        # Set up the gridspec with the desired proportions
        gs_main = gridspec.GridSpec(6, 1)
        gs_inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[:3, :], width_ratios=[9, 1], wspace=0.1)

        # Create the three axes with the specified vertical extensions
        ax1 = plt.subplot(gs_inner[0, 0])  # Top axis taking 3/6 of the vertical space, with space for the colorbar
        cax1 = plt.subplot(gs_inner[0, 1]) # Colorbar axis next to ax1
        ax2 = plt.subplot(gs_main[3:5, :])  # Middle axis taking 2/6 of the vertical space
        ax3 = plt.subplot(gs_main[5, :])    # Bottom axis taking 1/6 of the vertical space

        return fig, (ax1, cax1, ax2, ax3)

    def plot_spike_occurrences(self, spike_df, ax):
        """
        Plot spike occurrences as vertical lines on the given axis.

        Parameters
        ----------
        spike_df : pandas.DataFrame
            A DataFrame containing spike times in the 'spike_peak_s' column.
        ax : matplotlib.axes.Axes
            The Axes object to draw the spike occurrences on.
        """
        # Set the x-axis limits
        ax.set_xlim(0, 5)

        # Iterate through the spike times and plot a short vertical line for each
        for spike_time in spike_df['spike_peak_s']:
                ax.axvline(x=spike_time, ymin=0.45, ymax=0.55, linewidth=1, color='k')

        # Set axis labels
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Spike Occurrences')
        ax.set_yticks([])  # Remove y-axis ticks as they are not relevant in this plot

    def plot_two_parameters(self, fig, ax, timeAx, param1, param2, param1_label, param2_label):
        """
        Plot two parameters on a single plot with two y-axes.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The Figure object to draw on.
        ax : matplotlib.axes.Axes
            The Axes object to draw the first parameter data on.
        timeAx : array-like
            The time axis for the plot.
        param1 : array-like
            The data for the first parameter.
        param2 : array-like
            The data for the second parameter.
        param1_label : str
            The label for the first parameter data.
        param2_label : str
            The label for the second parameter data.
        """
        color = 'tab:blue'
        ax.plot(timeAx, param1, color=color)
        ax.set_xlabel('time, s')
        ax.set_ylabel(param1_label, color=color)
        ax.tick_params(axis='y', labelcolor=color)
        ax.set_xlim((time_ax[0],time_ax[-1]))
        ax2 = ax.twinx()

        color = 'xkcd:sky blue'
        ax2.plot(timeAx, param2, color=color)
        ax2.set_ylabel(param2_label, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

    def plot_contours(self, ax, cax, traceContour, fps, num_contours=200, colormap='viridis', alpha=0.5, outline=True):
        """
        Plot the contours with translucent patches.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The Axes object to draw the contours on.
        cax : matplotlib.axes.Axes
            The Axes object to draw the colorbar on.
        traceContour : list of lists
            A list of lists containing the x and y coordinates of the polygons.
        fps : float
            The frame rate of the video.
        num_contours : int, optional
            The number of contours to plot, linearly spaced throughout the list. (default: 200)
        colormap : str, optional
            The colormap to use for the patches. (default: 'viridis')
        alpha : float, optional
            The transparency level for the patches. (default: 0.5)
        outline : bool, optional
            Whether to draw a black outline around the patches. (default: True)
        """
        # Generate indices for linearly spaced contours
        contour_indices = np.linspace(0, len(traceContour) - 1, num_contours, dtype=int)

        # Get the colormap
        cmap = plt.get_cmap(colormap)
        # Find the axis limits
        min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')

        # Plot the contours
        for i, idx in enumerate(contour_indices):
            contour = np.array(traceContour[idx])
            # Smooth the contour
            num_points = len(contour)
            t = np.linspace(0, 1, num_points)
            new_t = np.linspace(0, 1, num_points * 5)  # Increase the number of points for a smoother contour

            x_spline = make_interp_spline(t, contour[:, 0], k=3)(new_t)
            y_spline = make_interp_spline(t, contour[:, 1], k=3)(new_t)
            smoothed_contour = np.column_stack((x_spline, y_spline))

            polygon = patches.Polygon(
                smoothed_contour,
                closed=True,
                facecolor=cmap(i / len(contour_indices)),
                alpha=alpha,
                edgecolor='black' if outline else None
            )
            ax.add_patch(polygon)

            # Update axis limits
            min_x, min_y = np.minimum((min_x, min_y), contour.min(axis=0))
            max_x, max_y = np.maximum((max_x, max_y), contour.max(axis=0))

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_aspect('equal')

        # Add colorbar
        norm = plt.Normalize(vmin=0, vmax=(len(traceContour) - 1) / fps)
        cbar = ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
        cbar.set_label('Time (s)')    
        
    def create_final_plot(self, spike_df, time_ax, trace, interp_instant_freq, traceContour, fps):
        """
        Create a final plot that combines spike occurrences, two parameters, and contours in a single figure.

        Parameters
        ----------
        spike_df : pandas.DataFrame
            A DataFrame containing spike times in the 'spike_peak_s' column.
        time_ax : array-like
            The time axis for the plot.
        trace : array-like
            The trace data.
        interp_instant_freq : array-like
            The interpolated instantaneous frequency data.
        traceContour : list of lists
            A list of lists containing the x and y coordinates of the polygons.
        fps : float
            The frame rate of the video.

        Returns
        -------
        None
        """

        f, ax_list = self.create_vertical_axes()
        self.plot_spike_occurrences(spike_df, ax_list[3])
        self.plot_two_parameters(f, ax_list[2], time_ax, np.abs(trace[:, 3]), interp_instant_freq,
                                 'thrust, m/s', 'instant. spike frequency, Hz')
        self.plot_contours(ax_list[0], ax_list[1], traceContour, fps, num_contours=200, colormap='viridis', alpha=0.5)

        f.tight_layout()
        plt.show()
