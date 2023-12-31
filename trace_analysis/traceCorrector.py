import numpy as np
import matplotlib.pyplot as plt
from data_handlers.matLabResultLoader import matLabResultLoader
from data_handlers.mediaHandler import mediaHandler
from plotting.fishPlot import frameOverlay
class traceCorrector:

    def __init__(self,dataDict):
        """
        Initializes the traceCorrector with the provided data dictionary.

        Args:
            dataDict (dict): A dictionary containing metadata and file positions.
        """
        # dictionary with the meta data and file positions
        self.dataDict = dataDict


        # read arena box
        if self.dataDict['csv'] == '':
            self.mmTraceAvailable = True
        else:
            self.mmTraceAvailable = False
            self.boxCoords = np.genfromtxt(self.dataDict['csv'],delimiter=',')      

        # load matlab data
        self.matLabLoader = matLabResultLoader(self.dataDict['anaMat'])
        self.matLabLoader.getData()
        # load movie file
        if self.dataDict['seq'] != '':
            self.mH = mediaHandler(self.dataDict['seq'],'norpix')
        elif self.dataDict['avi'] != '':
            self.mH = mediaHandler(self.dataDict['avi'],'movie')
        
        #shorthands
        self.contour          = self.matLabLoader.traceContour
        self.head             = self.matLabLoader.traceHead
        self.tail             = self.matLabLoader.traceTail
        self.midLine          = self.matLabLoader.traceMidline
        if self.mH.modus == 'norpix':
            self.headerDict       = self.mH.media.header_dict
            self.originFrame      = self.headerDict ['origin']
            self.allocated_frames = self.headerDict['allocated_frames'] 
            self.fps              = self.headerDict['suggested_frame_rate'] 
        elif self.mH.modus == 'movie':
            self.headerDict       = None
            self.originFrame      = 0
            self.allocated_frames = self.mH.length 
            self.fps              = self.mH.fps

        self.pixelOffset      = np.array([0.,0.])
        
        #preallocations
        self.currentFrame = None
        self.frameI       = 0
        
        # calibration
        self.calibrationOngoing = False
        self.frameShift         = 0
        self.pixelOffset        = np.array([0.,0.])
        self.coordShift         = np.zeros(shape=(1,2))
        
        #matplotlib 
        self.fig,self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)

    def close_figure(self):
        """
        Closes the current figure displayed by the traceCorrector.
        """
        plt.close(self.fig)

    def calculateCoordShift(self,bufferShift):
        """
        Calculates the coordinate shift based on the provided buffer shift value.

        Args:
            bufferShift (float): The buffer shift value.

        Returns:
            numpy.ndarray: The calculated coordinate shift.
        """
        imageWidth = self.mH.imageWidth # shorthand
        xShift = int(bufferShift%imageWidth)  # modulo
        yShift = bufferShift//imageWidth # integer division
        return np.array([xShift,yShift])

    def shiftFrameCoords(self):
        """
        Applies the calculated coordinate shift to the tracking data.
        """
        self.head    = self.head    + self.coordShift
        self.tail    = self.tail    + self.coordShift
        self.contour = [x + self.coordShift for x in self.contour]
        self.midLine = [x + self.coordShift for x in self.midLine]

    def plotFrameOverlay(self):
        """
        Plots the current frame with the tracking data overlayed.
        """
        frameOverlay(self.ax, self.currentFrame, self.contour[self.frameI],
                     self.midLine[self.frameI], self.head[self.frameI,:], 
                     self.tail[self.frameI,:], self.boxCoords)
        if self.calibrationOngoing:
            self.ax.set_title('q = quit | f = fullscreen | a = -1 frame | A -10 frames | c = +1 frame | D +10 frames | w = negative origin | e = origin frame| cursor moves detection | s = save frame' )
            self.ax.set_xlabel('frame offSet: ' + str(self.frameShift) + ' | origin frame: '+ str(self.originFrame) + ' | pixelShift (x,y): ' + str(self.pixelOffset) )
        else:
            self.ax.set_xlabel('frame: ' + str(self.frameI) + ' | dur: '+ str(np.round(self.frameI/self.fps,2)))
        plt.draw()

    def getFrameNo4Norpix(self,correctionShift):
        """
        Returns the corrected frame number for Norpix files.

        Args:
            correctionShift (int): The correction shift value.

        Returns:
            int: The corrected frame number.
        """
        # negative shifts are set as a positive shift with AllocatedFrame - shift
        if correctionShift < 0:
            correctionShift = self.allocated_frames + correctionShift
        # return corrected frame number
        return int(np.abs(((self.frameI+self.originFrame+correctionShift)%self.allocated_frames)))

    def loadNorPixFrame(self,frameShift):
        """
        Loads the frame from the Norpix file based on the provided frame shift value.

        Args:
            frameShift (int): The frame shift value.

        Returns:
            numpy.ndarray: The loaded frame.
        """
        return self.mH.getFrame(self.getFrameNo4Norpix(frameShift))

    def on_press(self,event):
        """
        Handles key press events for the user interface.

        Args:
            event (matplotlib.backend_bases.KeyEvent): The key press event.
        """
        shiftCoord = False 
        loadNewImg = False
        if event.key == 'a':
            self.frameShift -=1
            loadNewImg = True
        elif event.key == 'd':
            self.frameShift +=1
            loadNewImg = True
        elif event.key == 'A':
            self.frameShift -=10
            loadNewImg = True
        elif event.key == 'D':
            self.frameShift +=10
            loadNewImg = True
        elif event.key =='w':
            self.frameShift = self.originFrame*-1
            loadNewImg = True
        elif event.key =='e':
            self.frameShift = 0
            loadNewImg = True
        elif event.key == 'q' or event.key == 'Q':
            self.calibrationOngoing = False
        elif event.key == 'right':
            self.coordShift = np.array([1,0])
            self.pixelOffset[0] +=1
            shiftCoord = True
        elif event.key == 'left':
            self.coordShift = np.array([-1,0])
            self.pixelOffset[0] -=1
            shiftCoord = True
        elif event.key == 'up':
            self.coordShift = np.array([0,-1])
            self.pixelOffset[1] -=1
            shiftCoord = True
        elif event.key == 'down':
            self.coordShift = np.array([0,1])
            self.pixelOffset[1] +=1
            shiftCoord = True
        else:
            shiftCoord = False

        if shiftCoord:
            self.shiftFrameCoords()
        if self.calibrationOngoing:
            self.refreshImage(loadNewImg)


    def refreshImage(self, newImgFlag):
        """
        Refreshes the displayed image, either by loading a new frame or by redrawing
        the existing frame with updated tracking data.

        Args:
            newImgFlag (bool): True if a new frame should be loaded, False otherwise.
        """
        if newImgFlag:
            self.currentFrame = self.loadNorPixFrame(self.frameShift)
        plt.cla()
        self.plotFrameOverlay()
        self.fig.canvas.draw()

    def calibrateTracking(self):
        """
        Calibrates the tracking data by displaying an interactive interface for the user
        to adjust frame and coordinate shifts.
        """
        # set up figure
        self.frameI = 0
        self.calibrationOngoing = True
        self.refreshImage(True)
        plt.show()
    
    def runTest(self,lengthInFrames =100):
        """
        Runs a test by displaying a series of frames with the tracking data overlayed.

        Args:
            lengthInFrames (int, optional): The number of frames to display during the test. Defaults to 100.
        """
        self.fig,self.ax = plt.subplots()
        plt.ion()
        for frameI in np.linspace(0,self.allocated_frames-1,lengthInFrames, dtype=int ):
            self.frameI = frameI
            self.refreshImage(True)
            plt.pause(0.001)



