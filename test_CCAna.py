from importlib import reload
from counterCurrentAna import sortMultiFileFolder, matLabResultLoader
import numpy as np
from mediaHandler import mediaHandler


#reload(sortMultiFileFolder)

mff = sortMultiFileFolder('/media/gwdg-backup/BackUp/Vranda/data_counter_c-start/countercurrent_onefolder/rei_last_generation_11-2018') 
fileDict = mff.__main__() 

dataDict = fileDict['HMM2']

# read arena box
boxCoords = np.genfromtxt(dataDict['csv'],delimiter=',')      

# load matlab data
matLabLoader = matLabResultLoader(dataDict['anaMat'])
matLabLoader.getData()
# load movie file
mH = mediaHandler(dataDict['seq'],'norpix')


def calculateCoordShift(imageWidth,bufferShift):
    xShift = int(bufferShift%imageWidth)  # modulo
    yShift = bufferShift//imageWidth # integer division
    return np.array([xShift,yShift])

def shiftFrameCoords(head,tail,contour,midLine,coordShift):
    head = head + coordShift
    tail = tail + coordShift
    contour = contour + coordShift
    midLine = midLine + coordShift
    return head, tail, contour, midLine


def plotFrameOverlay(frame,midLine,head,tail,contour,boxCoords,coordShift):
    head, tail, contour, midLine = shiftFrameCoords(head,tail,contour,midLine,coordShift)
    plt.imshow(frame,cmap='gray')  
    #plt.hold(True)
    plt.plot(  midLine[:,0],midLine[:,1],'g.-')
    plt.plot(       head[0],     head[1],'bo')
    plt.plot(       tail[0],     tail[1],'bs')
    plt.plot(  contour[:,0],contour[:,1],'y-')
    plt.plot(boxCoords[:,0],boxCoords[:,1],'y-')
    plt.plot(boxCoords[[0,-1],0],boxCoords[[0,-1],1],'y-')
    plt.draw()

def getFrameNo4Norpix(headerDict,frameNo,bonus):
    return int(((frameNo+headerDict['origin'])%headerDict['allocated_frames'])+bonus)

def loadNorPixFrame(mH,frameI,frameShift):
    return mH.getFrame(getFrameNo4Norpix(mH.media.header_dict,frameI,frameShift))

import matplotlib.pyplot as plt  

contour = matLabLoader.traceContour
head    = matLabLoader.traceHead
tail    = matLabLoader.traceTail
midLine = matLabLoader.traceMidline

frameI  = 13258
pixelShift = np.array([3,-12])
frameShift = 0


plt.ion()

frame = loadNorPixFrame(mH,frameI,frameShift)
plt.cla()
plotFrameOverlay(frame,midLine[frameI],head[frameI,:],tail[frameI,:],contour[frameI],boxCoords,pixelShift)
plt.pause(0.001)

for frameI  in range(13000,14000):
    frame = loadNorPixFrame(mH,frameI,frameShift)
    plt.cla()
    plotFrameOverlay(frame,midLine[frameI],head[frameI,:],tail[frameI,:],contour[frameI],boxCoords,pixelShift)
    plt.pause(0.001)
