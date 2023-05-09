import matplotlib.pyplot as plt

class FishHabituationProfiler:
     
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
        """
        Creates a line plot for the Top_fraction over Day_number for each fish separately,
        with individual males in blue and females in red, based on the Tank_number, ID, and Sex columns in the DataFrame df.

        Args:
            df (pandas.DataFrame): The DataFrame to use for plotting.
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
