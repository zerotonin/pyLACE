import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
import pandas as pd

def frameOverlay(ax,frame,contour,midLine,head,tail,boxCoords,
                frameCmap = 'gray'):
    """
    Overlays the frame with trace results including contour, midline, head, tail, and box coordinates.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object to draw on.
    frame : array-like
        The frame to be displayed as the background.
    contour : array-like
        The contour points to be plotted.
    midLine : array-like
        The midline points to be plotted.
    head : array-like
        The head point to be plotted.
    tail : array-like
        The tail point to be plotted.
    boxCoords : array-like
        The bounding box coordinates to be plotted.
    frameCmap : str, optional
        The colormap to be used for the frame, default is 'gray'.
    """
    ax.imshow(frame,cmap=frameCmap)  
    plotTraceResult(ax,contour,midLine,head,tail,boxCoords)

def plotTraceResult(ax,contour,midLine,head,tail,boxCoords):
    """
    Plots the trace results including contour, midline, head, tail, and box coordinates.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object to draw on.
    contour : array-like
        The contour points to be plotted.
    midLine : array-like
        The midline points to be plotted.
    head : array-like
        The head point to be plotted.
    tail : array-like
        The tail point to be plotted.
    boxCoords : array-like
        The bounding box coordinates to be plotted.
    """
    ax.plot(midLine[:,0],midLine[:,1],'g.-')
    ax.plot(contour[:,0],contour[:,1],'y-')
    ax.plot(head[0],head[1],'bo')
    ax.plot(tail[0],tail[1],'bs')
    if boxCoords is not None:
        ax.plot(boxCoords[:,0],boxCoords[:,1],'y-')
        ax.plot(boxCoords[[0,-1],0],boxCoords[[0,-1],1],'y-')

def simpleSpatialHist(ax,probDensity,cmap='PuBuGn'):
    """
    Plots a simple spatial histogram on the given Axes object.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object to draw on.
    probDensity : array-like
        The probability density values to be plotted.
    cmap : str, optional
        The colormap to be used for the plot, default is 'PuBuGn'.
    """
    ax.imshow(probDensity,origin='lower',interpolation='gaussian',cmap=cmap)


def seabornSpatialHist(midLine):
    """
    Plots a spatial histogram using seaborn's JointGrid and kdeplot.

    Parameters
    ----------
    midLine : array-like
        The midline points to be plotted.
    """
    allMidLine =  np.vstack((midLine[:]))
    df = pd.DataFrame(data={'x-coordinate, mm' : allMidLine[:,0],'y-coordinate, mm':allMidLine[:,1]})

    sns.set_theme(style="white")
    cmap = sns.cubehelix_palette(start=1.66666, light=1, as_cmap=True)

    g = sns.JointGrid(data=df, x="x-coordinate, mm", y="y-coordinate, mm", space=0)
    g.plot_joint(sns.kdeplot,fill=True,cmap=cmap)
    g.ax_joint.set_aspect('equal')
    g.plot_marginals(sns.histplot, color="#173021", alpha=.75, bins=25)


def addColorBar(ax,cmap,vmin,vmax,orientation,axisLableStr):
    """
    Adds a colorbar to the given Axes object with specified properties.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object to draw on.
    cmap : str
        The colormap to be used for the colorbar.
    vmin : float
        The minimum value for the colorbar.
    vmax : float
        The maximum value for the colorbar.
    orientation : str
        The orientation of the colorbar, either 'h' for horizontal or 'v' for vertical.
    axisLableStr : str
        The label for the colorbar axis.
    """
    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    if orientation == 'h':
        cbar = plt.colorbar(sm,orientation="horizontal",ax=ax)
        cbar.ax.set_xlabel(axisLableStr, rotation=0)
    if orientation == 'v':
        cbar = plt.colorbar(sm,orientation="vertical",ax=ax)
        cbar.ax.set_xlabel(axisLableStr, rotation=90)


def midLinePlot(ax,traceMidline,start,stop,step,colormapStr,fps):
    """
    Plots the midline traces with a colormap indicating time progression.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object to draw on.
    traceMidline : array-like
        The midline points to be plotted.
    start : int
        The start index for the midline traces.
    stop : int
        The stop index for the midline traces.
    step : int
        The step size between midline traces.
    colormapStr : str
        The colormap to be used for the plot.
    fps : float
        The frames per second of the video.
    """

    cmap = cm.get_cmap(colormapStr)
    for i  in range(start,stop):
        if i%step == 0:
            midLine = traceMidline[i]
            c = (i-start)/ (stop-start)
            ax.plot(midLine[:,0],midLine[:,1],'.-',color=cmap(c))
            ax.plot(midLine[-1,0],midLine[-1,1],'k.')

    addColorBar(ax,cmap,0,(stop-start)/fps,'h','time, s')

    plt.gca().set_aspect('equal', adjustable='box')


def makeTimeAxis(length,fps,unit='s'):
    """
    Creates a time axis with specified length, frames per second, and unit.

    Parameters
    ----------
    length : int
        The length of the time axis.
    fps : float
        The frames per second of the video.
    unit : str, optional
        The time unit for the axis, default is 's' for seconds. Options are 's', 'ms', 'min', and 'h'.

    Returns
    -------
    time_axis : numpy.ndarray
        The time axis with the specified unit.
    """
    time_s = np.linspace(0,length/fps,length)
    if unit == 's':
        return time_s
    if unit == 'ms':
        return time_s*1000
    if unit == 'min':
        return time_s/60
    if unit == 'h':
        return time_s/3600



def plotAngleVelAbs(fig,ax,timeAx,angleDeg,velDegS,angleStr):
    """
    Plots the absolute angle and angular velocity on a single plot with two y-axes.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The Figure object to draw on.
    ax : matplotlib.axes.Axes
        The Axes object to draw the angle data on.
    timeAx : array-like
        The time axis for the plot.
    angleDeg : array-like
        The angle data in degrees.
    velDegS : array-like
        The angular velocity data in degrees per second.
    angleStr : str
        The label for the angle data.
    """

    color = 'tab:blue'
    ax.set_xlabel('time, s')
    ax.set_ylabel(f'{angleStr} angle, deg', color=color)
    ax.plot(timeAx,angleDeg, color=color)
    ax.tick_params(axis='y', labelcolor=color)

    ax2 = ax.twinx()

    color ='xkcd:sky blue'
    ax2.set_ylabel(f'{angleStr} velocity, deg*s-1', color=color)
    ax2.plot(timeAx,velDegS, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout() 