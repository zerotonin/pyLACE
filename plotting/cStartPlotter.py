import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from scipy.interpolate import make_interp_spline
from matplotlib import patches
from data_handlers.mediaHandler import mediaHandler
import cv2
from tqdm import tqdm
class cStartPlotter:
    """
    A class for plotting the contours, spike occurrences, and two parameters in a single figure.
    Used for Garag et al 2023 A and B
    """
    def __init__(self) -> None:
         pass
         
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
        gs_inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[:3, :], width_ratios=[9, 1], wspace=0.01)

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

    def plot_two_parameters(self, fig, ax, time_ax, param1, param2, param1_label, param2_label,x_lim = None):
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
        if not x_lim:
            x_lim = ((time_ax[0],time_ax[-1]))
            

        color = 'tab:blue'
        ax.plot(time_ax, param1, color=color)
        ax.set_xlabel('time, s')
        ax.set_ylabel(param1_label, color=color)
        ax.tick_params(axis='y', labelcolor=color)
        ax.set_xlim(x_lim)
        ax2 = ax.twinx()
        ax.set_yscale('log')

        color = 'xkcd:sky blue'
        ax2.plot(time_ax, param2, color=color)
        ax2.set_ylabel(param2_label, color=color)
        ax2.set_ylim((0,2.5))
        ax2.tick_params(axis='y', labelcolor=color)

    def plot_contours(self, ax, cax, traceContour, fps, num_contours=200, 
                      colormap='viridis', alpha=0.5, outline=True, background_image=None,
                      contour_offset = None):
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
        background_image : array-like, optional
            The background image to display behind the contours. (default: None)
        contour_offset : tuple of float, optional
            The x and y offsets for the contours. (default: None)
        """

        # Display the background image, if provided
        if background_image is not None:
            ax.imshow(background_image, zorder=0)

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

            if contour_offset is None:
                x_spline = make_interp_spline(t, contour[:, 0], k=3)(new_t) 
                y_spline = make_interp_spline(t, contour[:, 1], k=3)(new_t)
            else:
                x_spline = make_interp_spline(t, contour[:, 0], k=3)(new_t) + contour_offset[0]
                y_spline = make_interp_spline(t, contour[:, 1], k=3)(new_t) + contour_offset[1]
            smoothed_contour = np.column_stack((x_spline, y_spline))

            polygon = patches.Polygon(
                smoothed_contour,
                closed=True,
                facecolor=cmap(i / len(contour_indices)),
                alpha=alpha,
                edgecolor='black' if outline else None
            )
            ax.add_patch(polygon)


        ax.set_xlim(0, 280)
        ax.set_ylim(0, 130)
        ax.set_aspect('equal')

        # Turn off the tick marks
        ax.set_xticks([])
        ax.set_yticks([])

        # Add colorbar
        norm = plt.Normalize(vmin=0, vmax=(len(traceContour) - 1) / fps *1000)
        cbar = ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
        cbar.set_label('Time (ms)')    
        
    def create_final_plot(self, spike_df, time_ax, trace, interp_instant_freq, traceContour, fps,background_image=None):
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
        f : matplotlib.figure.Figure
            The Figure object.
        ax1, cax1, ax2, ax3 : tuple of matplotlib.axes.Axes
            The Axes objects for the created subplots.
        background_image : array-like, optional
            The background image to display behind the contours. (default: None)
        """

        f, ax_list = self.create_vertical_axes()
        self.plot_spike_occurrences(spike_df, ax_list[3])
        self.plot_two_parameters(f, ax_list[2], time_ax, interp_instant_freq, np.abs(trace[:, 3]),
                                  'instant. spike frequency, Hz', 'thrust, m/s')
        self.plot_contours(ax_list[0], ax_list[1], traceContour, fps, num_contours=200, colormap='viridis', alpha=0.5,background_image=background_image)

        return f,ax_list
    
    def create_animated_plot(self, spike_df, time_ax, trace, interp_instant_freq, traceContour, 
                             fps, path_to_mediaFile,animation_file_position, comets_tail = 25, contour_offSet = None,
                             round_robin_offset = None):
        """
        Create an animated plot with gradual data filling and contour plot for the last 50 frames.

        Parameters are the same as `create_final_plot`, except for:
        background_images : list of array-like
            A list of background images to display behind the contours, one for each frame.
        """

        plt.rcParams.update({'font.size':8})
        # Prepare an empty list to store the frames
        frames = []
        mho = mediaHandler(path_to_mediaFile,'movie',fps,bufferSize = 2000)

        # Get the frame indices for the movie and contours
        frame_indices = self.get_frame_indices(time_ax, mho.fps)
        # Get the index to read oput video
        video_index = frame_indices
        if round_robin_offset != None:
            video_index = np.roll(video_index,round_robin_offset)


        # Loop over all frames
        for frame in tqdm(range(len(time_ax)),desc='Making frames'):
        #for frame in tqdm(range(2000,2100),desc='Making frames'):
            
            current_frame_index = frame_indices[frame]
            if current_frame_index >= comets_tail:
                try:
                    # Prepare Video data
                    current_traceContour = traceContour[current_frame_index-comets_tail:current_frame_index]
                    current_background_image = mho.getFrame(int(video_index[frame]))

                    # Prepare the data for this frame
                    current_spike_df = spike_df[spike_df['spike_peak_s'] <= time_ax[frame]]
                    current_trace = trace[:frame+1]
                    current_interp_instant_freq = interp_instant_freq[:frame+1]

                    # Create the plot for this frame
                    f, ax_list = self.create_vertical_axes()
                    self.plot_spike_occurrences(current_spike_df, ax_list[3])
                    ax_list[3].spines['right'].set_visible(False)
                    ax_list[3].spines['left'].set_visible(False)
                    ax_list[3].spines['top'].set_visible(False)
                    ax_list[3].set_yticks([])
                    ax_list[3].set_ylabel('Spikes')  # remove x-axis label

                    self.plot_two_parameters(f, ax_list[2], time_ax[:frame+1], current_interp_instant_freq, np.abs(current_trace[:, 3]),
                                            'instant. spike frequency, Hz', 'thrust, m/s',x_lim=(time_ax[0],time_ax[-1]))
                    
                    ax_list[2].set_xticks([])
                    ax_list[2].set_xlabel('')  # remove x-axis label
                    ax_list[2].set_xticklabels([])  # remove xtick labels

                    self.plot_contours(ax_list[0], ax_list[1], current_traceContour, fps, num_contours=200, colormap='viridis', 
                                       alpha=0.025,background_image=current_background_image,contour_offset=contour_offSet)

                    #f.tight_layout()
                    # Convert the plot to an image and add it to the list of frames
                    f.set_size_inches(4.16, 3.35)
                    f.set_dpi(300)  # Add this line
                    f.canvas.draw()
                    image = np.frombuffer(f.canvas.tostring_rgb(), dtype='uint8').reshape(f.canvas.get_width_height()[::-1] + (3,))
                    frames.append(image)

                    # Close the figure to free up memory
                    plt.close(f)
                except:
                    pass

        # Create the animation
        animation = cv2.VideoWriter(animation_file_position, cv2.VideoWriter_fourcc(*'mp4v'), 25, frames[0].shape[:2][::-1])
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            animation.write(frame_bgr)

        animation.release()

        return animation

    def get_frame_indices(self, time_ax, fps):
        """
        Get the corresponding frame index for each point in time_ax.

        Parameters
        ----------
        time_ax : array-like
            The time axis for the plot.
        fps : float
            The frame rate of the video.

        Returns
        -------
        frame_indices : numpy.array
            An array of frame indices corresponding to the times in time_ax.
        """
        # Calculate the frame index for each time point
        frame_indices = (time_ax * fps).astype(int)

        return frame_indices
