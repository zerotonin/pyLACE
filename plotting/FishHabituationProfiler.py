import matplotlib.pyplot as plt

class FishHabituationProfiler:
    """A class for profiling fish habituation using a DataFrame.

    This class is used to generate habituation profile plots for fish
    using their behavioral data in a pandas DataFrame. The plots display
    line graphs of various measures over time, separated by sex, and
    highlight the habituation limit using an arrow.

    Attributes:
        df (pandas.DataFrame): The DataFrame containing the fish data.
        measures (list of str): A list of measures to plot.
        limits (list of float): A list of habituation limits for each measure.
        y_labels (list of str): A list of y-axis labels for the plots.
        y_limits (list of tuple): A list of tuples specifying the y-axis limits for each plot.
        habituation_direction (list of bool): A list of booleans indicating the direction of the habituation arrow.
    """
     
    def __init__(self,df,measures = ['Top_fraction', 'Bottom_fraction', 'Tigmotaxis_transition_freq', 'Distance_travelled_cm'],
                                    limits = [0.4,0.1,0.2,4500], 
                                    y_labels=['In top zone, 0->1', 'In bottom zone, 0->1', 'Tigmotaxis frequency, Hz', 'Distance travelled, cm'], 
                                    y_limits = [(0,1),(0,1),(0,1),(0,10000)],
                                    habituation_direction = [False,True,True,True]):
        self.df = df
        self.measures = measures
        self.limits = limits
        self.y_limits = y_limits
        self.y_labels = y_labels
        self.habituation_direction = habituation_direction
          

    def check_habituation(self):
        """Generates line plots of fish habituation profiles for each fish in the DataFrame.

        This method groups the input DataFrame by Tank_number, ID, and Sex and creates
        a 2x2 grid of line plots for each fish, displaying the specified measures
        over time. Habituation limits are indicated using arrows.

        Returns:
            list of matplotlib.figure.Figure: A list of figure handles for the generated plots.
            list of str: A list of strings with tank number and fish ID for each plot.
        """
        # Group the data by Tank_number, ID, and Sex
        groups = self.df.groupby(["Tank_number", "ID", "Sex"])

        figure_handles = list()
        fish_id_list = list()
        # Loop through the groups and plot the lines with the corresponding colors and markers
        for (tank_num, fish_id, sex), group in groups:
            color = "blue" if sex == "M" else "red"
            label = f"{tank_num}, {fish_id}"
            fig, axes = plt.subplots(2, 2,figsize=(12, 8))
            axes = axes.flatten()

            for i in range(4):
                if i == 0:
                     self.plot_line_graph(axes[i],group,self.measures[i],color,label,self.y_limits[i],fish_id,tank_num,self.limits[i],self.y_labels[i],self.habituation_direction[i])
                else:
                     self.plot_line_graph(axes[i],group,self.measures[i],color,label,self.y_limits[i],None,None,self.limits[i],self.y_labels[i],self.habituation_direction[i])

            # Show the plot
            figure_handles.append(fig)
            fish_id_list.append(f"tankNum_{tank_num}_ID_{fish_id}")
        return figure_handles, fish_id_list

    def plot_line_graph(self, ax, group, measure, color, label, y_limits, fish_id, tank_num, limit, y_label, arrow_down):
        """Generates a line plot of the specified measure for a single fish.
        This method creates a line plot of the specified measure for a single fish
        within the input DataFrame. The plot is added to the provided axis object
        with the specified color, label, y-axis limits, and habituation limit.
        Habituation limits are indicated using an arrow.

        Args:
            ax (matplotlib.axes.Axes): The axis object to plot the line graph on.
            group (pandas.DataFrame): The DataFrame containing the data for a single fish.
            measure (str): The measure to plot.
            color (str): The color of the line plot.
            label (str): The label for the line plot.
            y_limits (tuple): A tuple specifying the y-axis limits for the plot.
            fish_id (str): The fish ID to be displayed in the plot title.
            tank_num (str): The tank number to be displayed in the plot title.
            limit (float): The habituation limit for the plot
            y_label (str): The y-axis label for the plot.
            arrow_down (bool): A boolean indicating the direction of the habituation arrow.
        """
        ax.plot(group["Day_number"], group[measure], color=color, label=label, marker='o')

        ax.set_xlabel("Day number")
        ax.set_ylabel(y_label)
        if fish_id:
            ax.set_title(f"Fish ID: {fish_id}, Tank number: {tank_num}")
        ax.set_ylim(y_limits)

        ax.legend()
        ax.axhline(y=limit, color="gray", linestyle="--")

        if arrow_down:
            ax.annotate('', xy=(group["Day_number"].max() + 0.5, limit), xycoords='data',
                        xytext=(group["Day_number"].max() + 0.5, limit + 0.3 * (y_limits[1] - y_limits[0])),
                        textcoords='data', arrowprops=dict(arrowstyle="->"))
        else:
            ax.annotate('', xy=(group["Day_number"].max() + 0.5, limit), xycoords='data',
                        xytext=(group["Day_number"].max() + 0.5, limit - 0.3 * (y_limits[1] - y_limits[0])),
                        textcoords='data', arrowprops=dict(arrowstyle="->"))
