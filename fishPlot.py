import matplotlib.pyplot as plt

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

