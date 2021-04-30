from importlib import reload
import numpy as np
import pandas as pd
from mediaHandler import mediaHandler
from traceCorrector import traceCorrector
from counterCurrentAna import sortMultiFileFolder
import traceAnalyser
import fishPlot
import matplotlib.pyplot as plt
import fishRecAnalysis
import fishDataBase


multiFileFolder = '/media/gwdg-backup/BackUp/Vranda/data_counter_c-start/countercurrent_onefolder/rei_last_generation_11-2018'
db = fishDataBase.fishDataBase()
db.runMultiTraceFolder(multiFileFolder,'rei','CCur','11-2018')

'''

fileDict = mff.__main__() 
#print(fileDict.keys())
dataDict = fileDict['INTF2']
expString = 'CCurr'
genName = 'rei'

reload(fishRecAnalysis)
fRAobj= fishRecAnalysis.fishRecAnalysis(dataDict,genName,expString)
fRAobj.correctionAnalysis()
#fRAobj.makeSaveFolder()
#fRAobj.saveResults()
dbEntry = fRAobj.saveDataFrames()
'''