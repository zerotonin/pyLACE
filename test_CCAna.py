from importlib import reload
import numpy as np
from mediaHandler import mediaHandler
from traceCorrector import traceCorrector
from counterCurrentAna import sortMultiFileFolder
import traceAnalyser

#reload(sortMultiFileFolder)

mff = sortMultiFileFolder('/media/gwdg-backup/BackUp/Vranda/data_counter_c-start/countercurrent_onefolder/rei_last_generation_11-2018') 
fileDict = mff.__main__() 
print(fileDict.keys())
dataDict = fileDict['INTM7']

traCor = traceCorrector(dataDict)
traCor.calibrateTracking()
print("calibration done")
traCor.runTest()

traAna = traceAnalyser.traceAnalyser(traCor)
traAna.pixelTrajectories2mmTrajectories()