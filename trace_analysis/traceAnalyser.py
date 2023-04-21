
from sqlite3 import enable_shared_cache
import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d

class traceAnalyser():
    """
    A class for analyzing fish trace data from movies. The class takes traceCorrectorObj as input and processes the
    data to compute various parameters related to the fish's movements, position, and other characteristics.
    """

    def __init__(self,traceCorrectorObj,default_arena_size):
        """
        Initializes the traceAnalyser object with given traceCorrectorObj and initializes various attributes.

        Args:
            traceCorrectorObj: An object containing trace correction data.
        """
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
        self.arena_size_by_experiment = default_arena_size

        # arena coordinates 
        if self.mm_tra_available == False:
            self.arenaCoords_mm  = np.array([[0,0],[self.arena_size_by_experiment[1],0],
                                             [self.arena_size_by_experiment[1],self.arena_size_by_experiment[0]],
                                             [0,self.arena_size_by_experiment[1]]])
            self.arenaCoords_pix = traceCorrectorObj.boxCoords 
            self.sortCoordsArenaPix()
            self.makeInterpolator()
            self.yaw = traceCorrectorObj.matLabLoader.trace[:,2]
            
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
        self.medianDivergenceFromStraightInZone_DEG =None


    def exportMetaDict(self):
        """
        Returns a dictionary containing the metadata of the analyzed trace.

        Returns:
            dict: A dictionary containing metadata of the analyzed trace.
        """
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
        """
        Returns a dictionary containing the metadata of the analyzed trace.

        Returns:
            dict: A dictionary containing metadata of the analyzed trace.
        """
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
        """
        Returns a dictionary containing the metadata of the analyzed trace.

        Returns:
            dict: A dictionary containing metadata of the analyzed trace.
        """
        if self.frameOffset < 0:
            frameShift = self.frameOffset + self.traceLenFrame
        else:
            frameShift = self.frameOffset
            
        self.movieIDX = (np.arange(self.traceLenFrame)+self.originFrame + frameShift)%self.traceLenFrame

    def sortCoordsArenaPix(self):
        """
        Returns a dictionary containing the metadata of the analyzed trace.

        Returns:
            dict: A dictionary containing metadata of the analyzed trace.
        """
        descY = np.flipud(self.arenaCoords_pix[np.argsort(self.arenaCoords_pix[:, 1])])
        lowRow  = descY[0:2,:]
        highRow = descY[2::,:]
        self.arenaCoords_pix = np.vstack((lowRow[np.argsort(lowRow[:,0])],np.flipud(highRow[np.argsort(highRow[:,0])])))
    
    def makeInterpolator(self):
        """
        Returns a dictionary containing the metadata of the analyzed trace.

        Returns:
            dict: A dictionary containing metadata of the analyzed trace.
        """
        x = self.arenaCoords_pix[:,0]
        y = self.arenaCoords_pix[:,1]
        self.interpX = LinearNDInterpolator(list(zip(x, y)), self.arenaCoords_mm[:,0])
        self.interpY = LinearNDInterpolator(list(zip(x, y)), self.arenaCoords_mm[:,1])

    def interpolate2mm(self,coords2D):
        """
        Returns a dictionary containing the metadata of the analyzed trace.

        Returns:
            dict: A dictionary containing metadata of the analyzed trace.
        """
        return np.vstack((self.interpX(coords2D),self.interpY(coords2D))).T

    def pixelTrajectories2mmTrajectories(self):
        """
        Returns a dictionary containing the metadata of the analyzed trace.

        Returns:
            dict: A dictionary containing metadata of the analyzed trace.
        """
        
        self.head_mm    = self.interpolate2mm(self.head_pix) 
        self.tail_mm    = self.interpolate2mm(self.tail_pix) 
        self.contour_mm = [self.interpolate2mm(x) for x in self.contour_pix] 
        self.midLine_mm = [self.interpolate2mm(x) for x in self.midLine_pix] 

        if self.mm_tra_available == False:
            self.create_trace_mm_denovo()
            self.mm_tra_available = True
            
    def create_trace_mm_denovo(self):
        """
        Create the trace_mm array by averaging the head_mm and tail_mm arrays, and including yaw and speeds.
        The result is stored in the self.trace_mm attribute.
        """
        # Calculate the midpoint between head_mm and tail_mm
        trace_mm = (self.head_mm + self.tail_mm) / 2.0

        # Reshape the yaw array and concatenate it with trace_mm
        yaw = self.yaw.reshape(-1, 1)
        trace_mm = np.hstack((trace_mm, yaw))

        # Calculate the speeds by differentiating trace_mm along the first axis and multiplying by fps
        speeds = np.diff(trace_mm, axis=0) * self.fps

        # Add a row of NaNs at the end of the speeds array
        speeds = np.vstack((speeds, np.full((1, 3), np.nan)))
        speeds[:,2] = np.rad2deg(speeds[:,2])

        # Concatenate trace_mm and speeds arrays
        trace_mm = np.hstack((trace_mm, speeds))

        # Store the result in self.trace_mm
        self.trace_mm = trace_mm

    def calculateSpatialHistogram(self,bins=[16,8]):
        """
        Returns a dictionary containing the metadata of the analyzed trace.

        Returns:
            dict: A dictionary containing metadata of the analyzed trace.
        """
        if self.mm_tra_available:
            temp = np.histogram2d(self.trace_mm[:,1],self.trace_mm [:,0],bins,density=True) # matlab trajectories are x than y therefore we have to flip the inices here
        else:
            allMidLine =  np.vstack((self.midLine_mm[:]))
            temp = np.histogram2d(allMidLine[:,0],allMidLine[:,1],bins,density=True)
        self.probDensity  = temp[0].T
        self.probDensity_xCenters = temp[1]
        self.probDensity_yCenters = temp[2]


    def calculateInZoneIDX(self):
        """
        Returns a dictionary containing the metadata of the analyzed trace.

        Returns:
            dict: A dictionary containing metadata of the analyzed trace.
        """
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
        """
        Analyzes the fish's behavior within the specified zone, including the in-zone fraction, duration, bendability,
        and median divergence from a straight path.
        """
        self.calculateInZoneIDX()
        self.inZoneFraction = sum(self.zoneIDX)/self.traceLenFrame
        self.inZoneDuration = self.inZoneFraction*self.traceLenSec
        self.inZoneBendability = [i for indx,i in enumerate(self.bendability) if self.zoneIDX[indx] == True]
        self.medianDivergenceFromStraightInZone_DEG = np.median([np.sum(np.abs(x[:,1]-180)) for x in self.inZoneBendability])

    
    def calculateBodyLength(self,midLine):
        """
        Calculates the body length of the fish based on the given midLine data.

        Args:
            midLine (numpy.ndarray): A 2D array containing midLine coordinates.

        Returns:
            float: The body length of the fish.
            numpy.ndarray: A 1D array representing the body-length axis.
        """
        vectorNorms =np.linalg.norm(np.diff(midLine, axis = 0),axis=1)
        bodyLen = vectorNorms.sum()
        bodyAxis = np.cumsum(np.insert(vectorNorms,0,0.,axis =0))
        return bodyLen,bodyAxis
    
    def interpMidLine(self,midLine,step = 10):
        """
        Interpolates the given midLine data to create a new midLine with evenly spaced points along the body-length axis.

        Args:
            midLine (numpy.ndarray): A 2D array containing midLine coordinates.
            step (int, optional): The number of evenly spaced points along the body-length axis. Defaults to 10.

        Returns:
            numpy.ndarray: A 2D array containing the new interpolated midLine.
        """
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
        """
        Computes uniform midLines for both pixel-based and millimeter-based data (if available) with the specified
        number of midLine points.

        Args:
            midLinePoints (int, optional): The number of points in the uniform midLine. Defaults to 10.
        """
        self.midLineUniform_pix = self.get_uniform_midline_subroutine(self.midLine_pix,midLinePoints)
        if self.mm_tra_available == False:
            self.midLineUniform_mm = self.get_uniform_midline_subroutine(self.midLine_mm,midLinePoints)

        

    def get_uniform_midline_subroutine(self,mid_line,mid_line_points):
        """
        Helper function to compute the uniform midLines for either pixel-based or millimeter-based data.

        Args:
            mid_line (numpy.ndarray): A 2D array containing midLine coordinates in pixel or millimeter space.
            mid_line_points (int): The number of points in the uniform midLine.

        Returns:
            numpy.ndarray: A 3D array containing the uniform midLines for the input data.
        """
        mid_line_result = list()
        for mL in mid_line:
            mid_line_result.append(self.interpMidLine(mL,mid_line_points))
        # convert the list to
        return np.array(mid_line_result)

        




