import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def frameOverlay(ax,frame,contour,midLine,head,tail,boxCoords,
                frameCmap = 'gray'):
    ax.imshow(frame,cmap=frameCmap)  
    plotTraceResult(ax,contour,midLine,head,tail,boxCoords)

def plotTraceResult(ax,contour,midLine,head,tail,boxCoords):
    ax.plot(midLine[:,0],midLine[:,1],'g.-')
    ax.plot(contour[:,0],contour[:,1],'y-')
    ax.plot(head[0],head[1],'bo')
    ax.plot(tail[0],tail[1],'bs')
    ax.plot(boxCoords[:,0],boxCoords[:,1],'y-')
    ax.plot(boxCoords[[0,-1],0],boxCoords[[0,-1],1],'y-')

def simpleSpatialHist(ax,probDensity,cmap='PuBuGn'):
    ax.imshow(probDensity,origin='lower',interpolation='gaussian',cmap=cmap)


def seabornSpatialHist(midLine):
    allMidLine =  np.vstack((midLine[:]))
    df = pd.DataFrame(data={'x-coordinate, mm' : allMidLine[:,0],'y-coordinate, mm':allMidLine[:,1]})

    sns.set_theme(style="white")
    cmap = sns.cubehelix_palette(start=1.66666, light=1, as_cmap=True)

    g = sns.JointGrid(data=df, x="x-coordinate, mm", y="y-coordinate, mm", space=0)
    g.plot_joint(sns.kdeplot,fill=True,cmap=cmap)
    g.ax_joint.set_aspect('equal')
    g.plot_marginals(sns.histplot, color="#173021", alpha=.75, bins=25)

