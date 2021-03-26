from importlib import reload
import numpy as np
from mediaHandler import mediaHandler
from traceCorrector import traceCorrector
from counterCurrentAna import sortMultiFileFolder
import traceAnalyser
import fishPlot
import matplotlib.pyplot as plt

#reload(sortMultiFileFolder)

mff = sortMultiFileFolder('/media/gwdg-backup/BackUp/Vranda/data_counter_c-start/countercurrent_onefolder/rei_last_generation_11-2018') 
fileDict = mff.__main__() 
print(fileDict.keys())
dataDict = fileDict['INTF2']

traCor = traceCorrector(dataDict)
traCor.calibrateTracking()
print("calibration done")
#traCor.runTest()
reload(traceAnalyser)
reload(fishPlot)
traAna = traceAnalyser.traceAnalyser(traCor)
traAna.pixelTrajectories2mmTrajectories()
traAna.calculateSpatialHistogram()
traAna.inZoneAnalyse()

frameI = 1468

reload(fishPlot)
fig,axs = plt.subplots(2)
fishPlot.frameOverlay(axs[0],traCor.mH.getFrame(traAna.movieIDX[frameI]),traAna.contour_pix[frameI],
                      traAna.midLine_pix[frameI],traAna.head_pix[frameI,:],
                      traAna.tail_pix[frameI,:],traAna.arenaCoords_pix)
fishPlot.plotTraceResult(axs[1],traAna.contour_mm[frameI],
                      traAna.midLine_mm[frameI],traAna.head_mm[frameI,:],
                      traAna.tail_mm[frameI,:],traAna.arenaCoords_mm)


fig,axs = plt.subplots()
fishPlot.simpleSpatialHist(axs,traAna.probDensity)
fishPlot.seabornSpatialHist(traAna.midLine_mm)
plt.show()

