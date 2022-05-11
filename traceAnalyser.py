
from sqlite3 import enable_shared_cache
import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d

class traceAnalyser():

    def __init__(self,traceCorrectorObj):
        # some data already is completely analysed in MatLab than
        self.mm_tra_available = traceCorrectorObj.mmTraceAvailable

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
        self.traceLenFrame = traceCorrectorObj.allocated_frames
        self.originFrame   = traceCorrectorObj.originFrame
        self.fps           = traceCorrectorObj.fps
        self.traceLenSec   = self.traceLenFrame/self.fps
        self.makeMovieIDX()

        # meta data
        self.genotype = traceCorrectorObj.dataDict['genotype'] 
        self.sex      = traceCorrectorObj.dataDict['sex']
        self.animalNo = traceCorrectorObj.dataDict['animalNo']

        # arena coordinates 
        if self.mm_tra_available == False:
            self.arenaCoords_mm  = np.array([[0,0],[162,0],[162,43],[0,43]])
            self.arenaCoords_pix = traceCorrectorObj.boxCoords 
            self.sortCoordsArenaPix()
            self.makeInterpolator()
            self.trace_mm
        else:
            self.trace_mm = traceCorrectorObj.matLabLoader.trace
        self.zoneMargins  = np.array([[40,11.5],[163,31.5]])

        #preallocators
        self.exportDict           = traceCorrectorObj.dataDict
        self.inZoneFraction       = None
        self.inZoneDuration       = None  
        self.probDensity_xCenters = None 
        self.probDensity_yCenters = None        
        self.inZoneBendability    = None
        self.midLineUniform_mm    = None
        self.midLineUniform_pix   = None
        self.head_mm              = None    
        self.tail_mm              = None    
        self.contour_mm           = None 
        self.midLine_mm           = None 
        self.probDensity          = None 


    def exportMetaDict(self):
        # advance exportDict
        self.exportDict['movieFrameIDX']            = self.movieIDX
        self.exportDict['fps']                      = self.fps
        self.exportDict['traceLenFrame']            = self.traceLenFrame
        self.exportDict['traceLenSec']              = self.traceLenSec
        self.exportDict['inZoneFraction']           = self.inZoneFraction
        self.exportDict['inZoneDuration']           = self.inZoneDuration
        self.exportDict['inZoneMedDiverg_Deg']      = self.medianDivergenceFromStraightInZone_DEG
        self.exportDict['probDensity_xCenters']     = self.probDensity_xCenters
        self.exportDict['probDensity_yCenters']     = self.probDensity_yCenters
        self.exportDict['path2_inZoneBendability']  = None
        self.exportDict['path2_midLineUniform_mm']  = None
        self.exportDict['path2_midLineUniform_pix'] = None
        self.exportDict['path2_head_mm']            = None
        self.exportDict['path2_tail_mm']            = None
        self.exportDict['path2_probDensity']        = None
        
        return self.exportDict

    def exportDataList(self):
        self.dataList = list()
        if not isinstance(self.inZoneBendability,type(None)):
            self.dataList.append(['inZoneBendability', self.inZoneBendability,3])
        if not isinstance(self.midLineUniform_mm,type(None)):
            self.dataList.append(['midLineUniform_mm', np.array(self.midLineUniform_mm),3])
        if not isinstance(self.midLineUniform_pix,type(None)):
            self.dataList.append(['midLineUniform_pix',np.array(self.midLineUniform_pix),3])
        if not isinstance(self.head_mm,type(None)):
            self.dataList.append(['head_mm',self.head_mm,2])
        if not isinstance(self.tail_mm,type(None)):
            self.dataList.append(['tail_mm',self.tail_mm,2])
        if not isinstance(self.probDensity,type(None)):
            self.dataList.append(['probDensity',self.probDensity,2])
        if self.mm_tra_available == True:
            self.dataList.append(['trace_mm',self.trace_mm,2])

        return self.dataList

    def makeMovieIDX(self):
        if self.frameOffset < 0:
            frameShift = self.frameOffset + self.traceLenFrame
        else:
            frameShift = self.frameOffset
            
        self.movieIDX = (np.arange(self.traceLenFrame)+self.originFrame + frameShift)%self.traceLenFrame

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

    def calculateSpatialHistogram(self,bins=[16,8]):
        if self.mm_tra_available:
            temp = np.histogram2d(self.trace_mm[:,1],self.trace_mm [:,0],bins,density=True) # matlab trajectories are x than y therefore we have to flip the inices here
        else:
            allMidLine =  np.vstack((self.midLine_mm[:]))
            temp = np.histogram2d(allMidLine[:,0],allMidLine[:,1],bins,density=True)
        self.probDensity  = temp[0].T
        self.probDensity_xCenters = temp[1]
        self.probDensity_yCenters = temp[2]


    def calculateInZoneIDX(self):
        self.zoneIDX = list()
        for frameI in range(self.traceLenFrame):
            # shortHand
            mmt = self.trace_mm[frameI,:]
            if self. mm_tra_available:
                boolTests = [(mmt[0] >= self.zoneMargins[0,0]),
                             (mmt[1] >= self.zoneMargins[0,1]),
                             (mmt[0] <= self.zoneMargins[1,0]),
                             (mmt[1] <= self.zoneMargins[1,1])]
            
            else:
                mL = self.midLine_mm[frameI]
                # check if the whole body is inside the zone margins
                # all is true when all are true
                #    ... false when all are false            
                #    ... false when one is true and the rest false
                #    ... false when one is false and the rest true
                        
                boolTests = [(mL[:,0] >= self.zoneMargins[0,0]).all(),
                            (mL[:,1] >= self.zoneMargins[0,1]).all(),
                            (mL[:,0] <= self.zoneMargins[1,0]).all(),
                            (mL[:,1] <= self.zoneMargins[1,1]).all()]
            
            if all(boolTests):
                self.zoneIDX.append(True)
            else:
                self.zoneIDX.append(False)

    def inZoneAnalyse(self):
        self.calculateInZoneIDX()
        self.inZoneFraction = sum(self.zoneIDX)/self.traceLenFrame
        self.inZoneDuration = self.inZoneFraction*self.traceLenSec
        self.inZoneBendability = [i for indx,i in enumerate(self.bendability) if self.zoneIDX[indx] == True]
        self.medianDivergenceFromStraightInZone_DEG = np.median([np.sum(np.abs(x[:,1]-180)) for x in self.inZoneBendability])

    
    def calculateBodyLength(self,midLine):
        vectorNorms =np.linalg.norm(np.diff(midLine, axis = 0),axis=1)
        bodyLen = vectorNorms.sum()
        bodyAxis = np.cumsum(np.insert(vectorNorms,0,0.,axis =0))
        return bodyLen,bodyAxis
    
    def interpMidLine(self,midLine,step = 10):
        # get the bodylength and an axis along the bodylength
        bodyLen,bodyAxis = self.calculateBodyLength(midLine)

        # create interpolation functions for x and y
        fX = interp1d(bodyAxis,midLine[:,0],kind='cubic')
        fY = interp1d(bodyAxis,midLine[:,1],kind='cubic')

        # create ten evenly spaced points along the body-length-axis
        newBodyAxis = np.linspace(0,bodyAxis[-1],step)
 
        # interpolate the midLine at these points
        newX =fX(newBodyAxis)
        newY =fY(newBodyAxis)

        #return new midLine
        return np.vstack((newX,newY)).T
    
    def getUniformMidLine(self,midLinePoints =10):
        self.midLineUniform_pix = self.get_uniform_midline_subroutine(self.midLine_pix,midLinePoints)
        if self.mm_tra_available == False:
            self.midLineUniform_mm = self.get_uniform_midline_subroutine(self.midLine_mm,midLinePoints)

        

    def get_uniform_midline_subroutine(self,mid_line,mid_line_points):
        mid_line_result = list()
        for mL in mid_line:
            mid_line_result.append(self.interpMidLine(mL,mid_line_points))
        # convert the list to
        return np.array(mid_line_result)

        




