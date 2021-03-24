from importlib import reload
import numpy as np
from mediaHandler import mediaHandler
from traceCorrector import traceCorrector
from counterCurrentAna import sortMultiFileFolder

#reload(sortMultiFileFolder)

mff = sortMultiFileFolder('/media/gwdg-backup/BackUp/Vranda/data_counter_c-start/countercurrent_onefolder/rei_last_generation_11-2018') 
fileDict = mff.__main__() 

dataDict = fileDict['HMM1']

traCor = traceCorrector(dataDict)
traCor.calibrateTracking()
print("calibration done")
traCor.runTest()
traCor.