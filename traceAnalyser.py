
import numpy as np
from scipy.interpolate import LinearNDInterpolator
class traceAnalyser():

    def __init__(self,traceCorrectorObj):

        # fish data -> pixel based
        self.head_pix            = traceCorrectorObj.head 
        self.tail_pix            = traceCorrectorObj.tail 
        self.contour_pix         = traceCorrectorObj.contour 
        self.midLine_pix         = traceCorrectorObj.midLine 
        # fish data -> body based
        self.bendability      = traceCorrectorObj.matLabLoader.bendability
        self.binnedBend       = traceCorrectorObj.matLabLoader.binnedBend
        self.saccs            = traceCorrectorObj.matLabLoader.saccs
        self.trigAveSacc      = traceCorrectorObj.matLabLoader.trigAveSacc
        self.medMaxVelocities = traceCorrectorObj.matLabLoader.medMaxVelocities

        # movie data
        self.headerDict    = traceCorrectorObj.headerDict
        self.pixelOffset   = traceCorrectorObj.pixelOffset
        self.frameOffset   = traceCorrectorObj.frameShift
        self.traceLenFrame = traceCorrectorObj.headerDict['allocated_frames']
        self.fps           = traceCorrectorObj.headerDict['suggested_frame_rate']
        self.traceLenSec   = self.traceLenFrame/self.fps

        # meta data
        self.genotype = traceCorrectorObj.dataDict['genotype'] 
        self.sex      = traceCorrectorObj.dataDict['sex']
        self.animalNo = traceCorrectorObj.dataDict['animalNo']

        # arena coordinates 
        self.arenaCoords_mm  = np.array([[0,0],[162,0],[162,43],[0,43]])
        self.arenaCoords_pix = traceCorrectorObj.boxCoords 
        self.sortCoordsArenaPix()
        self.makeInterpolator()

    def sortCoordsArenaPix(self):
        descY = np.flipud(self.arenaCoords_pix[np.argsort(self.arenaCoords_pix[:, 1])])
        lowRow  = descY[0:2,:]
        highRow = descY[2::,:]
        self.arenaCoords_pix = np.vstack((lowRow[np.argsort(lowRow[:,0])],np.flipud(highRow[np.argsort(highRow[:,0])])))
    
    def makeInterpolator(self):
        x = self.arenaCoords_pix[:,0]
        y = self.arenaCoords_pix[:,1]
        self.interpX = LinearNDInterpolator(list(zip(x, y)), self.arenaCoords_mm[:,0])
        self.interpY = LinearNDInterpolator(list(zip(x, y)), self.arenaCoords_mm[:,1])

    def interpolate2mm(self,coords2D):
        return np.vstack((self.interpX(coords2D),self.interpY(coords2D))).T

    def pixelTrajectories2mmTrajectories(self):
        
        self.head_mm    = self.interpolate2mm(self.head_pix) 
        self.tail_mm    = self.interpolate2mm(self.tail_pix) 
        self.contour_mm = [self.interpolate2mm(x) for x in self.contour_pix] 
        self.midLine_mm = [self.interpolate2mm(x) for x in self.midLine_pix] 

    def calculateSpatialHistogram(self):
        pass

    def calculateBendingInCentralZone(self):
        pass

    def saveResults(self):
        pass
