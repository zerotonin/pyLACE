from importlib import reload
import numpy as np
from mediaHandler import mediaHandler
from traceCorrector import traceCorrector
from counterCurrentAna import sortMultiFileFolder
import traceAnalyser
import fishPlot
import matplotlib.pyplot as plt
import fishRecAnalysis

#reload(sortMultiFileFolder)

mff = sortMultiFileFolder('/media/gwdg-backup/BackUp/Vranda/data_counter_c-start/countercurrent_onefolder/rei_last_generation_11-2018') 
fileDict = mff.__main__() 
#print(fileDict.keys())
dataDict = fileDict['INTF2']
expString = 'CCurr'
genName = 'rei'

reload(fishRecAnalysis)
fRAobj= fishRecAnalysis.fishRecAnalysis(dataDict,genName,expString)
fRAobj.correctionAnalysis()
fRAobj.makeSaveFolder()


def save2DMatrix(tag,mat,dataDict):

        tag = fRAobj.dataList[3][0]
        mat = fRAobj.dataList[3][1]

        fileName = tag + '.txt'
        filePosition = os.path.join(fRAobj.savePath,fileName)
        np.savetxt(filePosition,mat)
        fRAobj.dataDict['path2_'+tag] = filePosition


'''
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
traAna.getUniformMidLine()
exportDict = traAna.exportMetaDict()


# Database Path are set like follows
# sourcePath / %experiment_%genotype_%sex_%animalNo_rec_%recNumber/
# each path holds:
# yaml file, image of test acquisition, image of spatial hist,
# text files of the matrices
expString = 'CCurr'
genName = 'rei'
databasePath = '/media/gwdg-backup/BackUp/Zebrafish/pythonDatabase'




frameI = 14222

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
'''