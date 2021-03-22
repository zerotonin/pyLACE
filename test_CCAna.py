from importlib import reload
from counterCurrentAna import sortMultiFileFolder
#reload(sortMultiFileFolder)

mff = sortMultiFileFolder('/media/gwdg-backup/BackUp/Vranda/data_counter_c-start/countercurrent_onefolder/rei_last_generation_11-2018') 
fileDict = mff.__main__() 


import numpy as np
import scipy.io
dataDict = fileDict['HMM2']

# read arena box
boxCoords = np.genfromtxt(dataDict['csv'],delimiter=',')      
# read matlab analysis
mat = scipy.io.loadmat(dataDict['anaMat'])
metaData = mat['metaData']
analysedData = mat['analysedData']


traceResult      = analysedData[0][0][0]

# traceInfo
#
# col  1: x-position in pixel
# col  2: y-position in pixel
# col  3: major axis length of the fitted ellipse
# col  4: minor axis length of the fitted ellipse
# col  5: ellipse angle in degree
# col  6: quality of the fit
# col  7: number of animals believed in their after final evaluation
# col  8: number of animals in the ellipse according to surface area
# col  9: number of animals in the ellipse according to contour length
# col 10: is the animal close to an animal previously traced (1 == yes)
# col 11: evaluation weighted mean
# col 12: detection quality [aU] if
# col 13: correction index, 1 if the area had to be corrected automatically
traceInfo        = traceResult[:.0]
traceContour     = traceResult[:,1]
traceMidline     = traceResult[:,2]
traceHead        = traceResult[:,3]
traceTail        = traceResult[:,4]

trace            = analysedData[0][0][1]
bendability      = analysedData[0][0][2]
binnedBend       = analysedData[0][0][3]
saccs            = analysedData[0][0][4]
trigAveSacc      = analysedData[0][0][5]
medMaxVelocities = analysedData[0][0][6] 
