from os import minor
import data_handlers.matLabResultLoader as matLabResultLoader
from importlib import reload
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import fishPlot,os
import numpy as np



reload(matLabResultLoader)
mrl = matLabResultLoader.matLabResultLoader('/media/dataSSD/zebraFischTrackingExample/2016_09_02__16_38_32_result_ana.mat')
traceInfo, traceContour, traceMidline, traceHead, traceTail, trace, bendability, binnedBend, saccs, trigAveSacc, medMaxVelocities = mrl.getData()

fig,ax = plt.subplots(1)
colormapStr = 'cividis'
fps = 200
step = 1
stop = 6000
start = 0
fishPlot.midLinePlot(ax,traceMidline,start,stop,step,colormapStr,fps)
plt.show()

mrl.traceResult



reload(fishPlot)
fig,ax = plt.subplots(3,1)
fps = 200
axT = fishPlot.makeTimeAxis(trace.shape[0],fps,'s')
ax[0].plot(axT, trace[:,3])
fishPlot.plotAngleVelAbs(fig,ax[1],axT,np.rad2deg(trace[:,2]),trace[:,5],'yaw')
bend = np.array([ np.mean(x[:,1],axis=0) for x in bendability])
ax[2].plot(axT, bend)

plt.show()


