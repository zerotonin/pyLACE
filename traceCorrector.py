import numpy as np
import matplotlib.pyplot as plt
from matLabResultLoader import matLabResultLoader
from mediaHandler import mediaHandler
from fishPlot import frameOverlay
class traceCorrector:

    def __init__(self,dataDict):
        # dictionary with the meta data and file positions
        self.dataDict = dataDict

        # self known arena sizes
        self.arena_sizes = {'cruise':(248,114),'c_start':(80,40),'counter_current':(167,45)}

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

    def calculateCoordShift(self,bufferShift):
        imageWidth = self.mH.imageWidth # shorthand
        xShift = int(bufferShift%imageWidth)  # modulo
        yShift = bufferShift//imageWidth # integer division
        return np.array([xShift,yShift])

    def shiftFrameCoords(self):
        self.head    = self.head    + self.coordShift
        self.tail    = self.tail    + self.coordShift
        self.contour = [x + self.coordShift for x in self.contour]
        self.midLine = [x + self.coordShift for x in self.midLine]

    def plotFrameOverlay(self):
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
        # negative shifts are set as a positive shift with AllocatedFrame - shift
        if correctionShift < 0:
            correctionShift = self.allocated_frames + correctionShift
        # return corrected frame number
        return int(np.abs(((self.frameI+self.originFrame+correctionShift)%self.allocated_frames)))

    def loadNorPixFrame(self,frameShift):
        return self.mH.getFrame(self.getFrameNo4Norpix(frameShift))

    def on_press(self,event):
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
        if newImgFlag:
            self.currentFrame = self.loadNorPixFrame(self.frameShift)
        plt.cla()
        self.plotFrameOverlay()
        self.fig.canvas.draw()

    def calibrateTracking(self):
        # set up figure
        self.frameI = 0
        self.calibrationOngoing = True
        self.refreshImage(True)
        plt.show()
    
    def runTest(self,lengthInFrames =100):
        self.fig,self.ax = plt.subplots()
        plt.ion()
        for frameI in np.linspace(0,self.allocated_frames-1,lengthInFrames, dtype=int ):
            self.frameI = frameI
            self.refreshImage(True)
            plt.pause(0.001)

    def interp_trace_mm(self,x_length,y_length,x_old,y_old):
        """ If the user entered the wrong dimensions of the arena and therefore
        wrongly calculated the mm trace this function can fix this. This

        WARNING the translational velocities will be approximations

        :param x_length: real x length of the arena
        :type x_length: float
        :param y_length: real y length of the arena
        :type y_length: float
        :param x_old: false x length of the arena
        :type x_old: float
        :param y_old: false y length of the arena
        :type y_old: float
        """
        pass

