import pims,tqdm,os
import numpy as np
from vidstab import VidStab, layer_overlay, download_ostrich_video
import cv2


class mediaHandler():
    """
    Class for handling media files such as movies, norpix sequences, and image sequences.
    """
    def __init__(self,filename,modus,fps=0,bufferSize = 2000):
        """
        Initializes a mediaHandler object.

        Args:
            filename (str): Name of the media file.
            modus (str): Mode of the media file, i.e. 'movie', 'norpix', or 'image'.
            fps (int, optional): Frames per second of the media file. Defaults to 0.
            bufferSize (int, optional): Maximum number of frames to buffer. Defaults to 2000.
        """
        self.activeFrame = []
        self.frameNo = 0
        self.modus = modus
        self.buffer = {}
        self.bufferLog = []
        self.bufferSize = bufferSize
        self.fileName = filename
        if (modus == 'movie'):
            self.media    = cv2.VideoCapture(filename)
            self.length   = self.media.get(cv2.CAP_PROP_FRAME_COUNT) 
            self.height   = self.media.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.width    = self.media.get(cv2.CAP_PROP_FRAME_WIDTH)
            if self.media.get(cv2.CAP_PROP_MONOCHROME)   == 0:
                self.colorDim = 3
            else:
                self.colorDim = 1

            self.fps    = self.media.get(cv2.CAP_PROP_FPS)   
            self.SR_makeIntParameters()
            
        elif(self.modus == 'norpix'):
            self.media = pims.NorpixSeq(filename)
            self.length = len(self.media)-1
            if len(self.media.frame_shape) ==2:
                self.height, self.width  = self.media.frame_shape
            else:
                self.height, self.width, self.colorDim = self.media.frame_shape
                    
            self.fps    = self.media.frame_rate
            self.SR_makeIntParameters()
        elif(self.modus == 'image'):
            # here the programs for image list should follow
            self.media =  pims.ImageSequence(filename)
            self.length = len(self.media)-1
            self.height, self.width, self.colorDim = self.media.frame_shape
            self.fps    = 25   
            self.SR_makeIntParameters()
        else:
            print('MediaHandler:unknown modus')
            
    def SR_makeIntParameters(self):
        """
        Converts float parameters to integers.
        """

        self.length = int(self.length)
        self.height = int(self.height)
        self.width  = int(self.width)
        self.size   = (self.width,self.height)

    def SR_setFrameNoInBounds(self,frameNo):
        """
        Sets the frame number within the bounds of the media file.

        Args:
            frameNo (int): Frame number.

        Returns:
            int: Frame number within the bounds of the media file.
        """
        if (frameNo <0):
            frameNo = 0
            #print 'frame No was below zero, now set to zero'
            
        elif (frameNo > self.length):
            frameNo = self.length
            #print 'frame No was larger than media length, now set to media length'

        return frameNo

    def getFrame(self,frameNo):
        """
        Returns the frame at the specified frame number.

        Args:
            frameNo (int): Frame number.

        Returns:
            numpy.ndarray: Frame at the specified frame number.
        """
        
        frameNo = self.SR_setFrameNoInBounds(frameNo)

        if (frameNo <0):
            frameNo = 0
            #print 'frame No was below zero, now set to zero'
            
        elif (frameNo > self.length):
            frameNo = self.length
            #print 'frame No was larger than media length, now set to media length'
            
        # check if frame can be read from buffer    
        if (frameNo in self.bufferLog): 
            self.activeFrame = np.array(self.buffer[frameNo], copy=True)
            self.frameNo     = frameNo
        else:
                
            if (self.modus == 'movie'):
                self.getFrameMov(frameNo)
            elif(self.modus == 'norpix'):
                self.getFrameNorpix(frameNo)
            elif(self.modus == 'image'):
                self.getFrameImage(frameNo)
            else:
                print('MediaHandler:unknown modus')
                
            #delete from buffer if to large
            if (len(self.bufferLog) > self.bufferSize):
                self.buffer.pop(self.bufferLog[0])
                self.bufferLog.pop(0)
                
            #update buffer
            self.buffer[frameNo] = np.array(self.activeFrame, copy=True)
            self.bufferLog.append(frameNo)
                            
        return self.activeFrame
            
    def get_frameNo(self):
        """
        Returns the current frame number.

        Returns:
            int: Current frame number.
        """
        return self.frameNo

    def getFrameMov(self,frameNo):
        """
        Gets a frame from a movie file.

        Args:
            frameNo (int): Frame number.

        Raises:
            Exception: If the frame is unreadable.

        Returns:
            numpy.ndarray: Frame at the specified frame number.
        """
        self.frameNo     = frameNo
        self.media.set(1,frameNo)
        flag,self.activeFrame = self.media.read(frameNo)   
        if not flag or self.activeFrame is None:
            raise Exception('Frame ' + str(frameNo) + ' unreadable in ' +self.fileName)
        #else:
        #    self.activeFrame = cv2.cvtColor( self.activeFrame, cv2.COLOR_BGR2RGB)
        
    def getFrameNorpix(self,frameNo):
        """
        Gets a frame from a norpix sequence.

        Args:
            frameNo (int): Frame number.

        Returns:
            numpy.ndarray: Frame at the specified frame number.
        """
        self.frameNo     = frameNo
        self.activeFrame = self.media.get_frame(frameNo)   
            
    def getFrameImage(self,frameNo):
        """
        Gets a frame from an image sequence.

        Args:
            frameNo (int): Frame number.

        Returns:
            numpy.ndarray: Frame at the specified frame number.
        """
        self.frameNo     = frameNo
        self.activeFrame = self.media.get_frame(frameNo)   
        
    
    def get_time(self):
        """
        Returns the time of the current frame.

        Returns:
            float: Time of the current frame.
        """
        return self.frameNo/self.fps
    
  
    
    def transcode_seq2avis(self,targetFile):
        if self.modus == 'norpix':
            # Get information about the norpix file
            sourceFPS = round(self.fps)
            frameShape = self.size 
            allocatedFrames = self.media.header_dict['allocated_frames']    

            # Define the codec and create VideoWriter object 
            fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
            out = cv2.VideoWriter(targetFile,fourcc, sourceFPS,frameShape) 

            for frameNo in tqdm.tqdm(range(allocatedFrames),desc='trasconding '+self.fileName):
                frame = self.getFrame(frameNo)
                gray_3c = cv2.merge([frame, frame, frame])
                out.write(gray_3c)
                cv2.imshow('frame',gray_3c)

            out.release()
        else:
            print('This function only works with norpix movie files')
    
    def register_movie(self,sourceFile, targetFile, border = 50):
        """
        Stabilizes and registers a movie file.

        Args:
            sourceFile (str): Name of the source movie file.
            targetFile (str): Name of the target stabilized and registered movie file.
            border (int, optional): Border size for stabilizing the movie. Defaults to 50.
        """
        if self.modus == 'movie':
            # Initialize object tracker, stabilizer, 
            object_tracker = cv2.TrackerCSRT_create()
            stabilizer     = VidStab()

            # Initialize bounding box for drawing rectangle around tracked object
            object_bounding_box = None
            
            # Define the codec and create VideoWriter object 
            fourcc     = cv2.VideoWriter_fourcc('X','V','I','D')
            sourceFPS  = round(self.fps)
            frameShape = (self.size[0]+2*border,self.size[1]+2*border)
            out        = cv2.VideoWriter(targetFile,fourcc, sourceFPS,frameShape)   

            while True:
                grabbed_frame, frame = self.media.read()

                # Pass frame to stabilizer even if frame is None
                stabilized_frame = stabilizer.stabilize_frame(input_frame=frame, border_size=border)
            
                # If stabilized_frame is None then there are no frames left to process
                if stabilized_frame is None:
                    break

                # Draw rectangle around tracked object if tracking has started
                if object_bounding_box is not None:
                    success, object_bounding_box = object_tracker.update(stabilized_frame)

                    if success:
                        (x, y, w, h) = [int(v) for v in object_bounding_box]
                        cv2.rectangle(stabilized_frame, (x, y), (x + w, y + h),
                                    (0, 255, 0), 2)

                # Display stabilized output
                cv2.imshow('Frame', stabilized_frame)
                out.write(stabilized_frame)

                key = cv2.waitKey(5)

                # Select ROI for tracking and begin object tracking
                # Non-zero frame indicates stabilization process is warmed up
                if stabilized_frame.sum() > 0 and object_bounding_box is None:
                    object_bounding_box = cv2.selectROI("Frame",
                                                        stabilized_frame,
                                                        fromCenter=False,
                                                        showCrosshair=True)
                    object_tracker.init(stabilized_frame, object_bounding_box)
                elif key == 27:
                    break

            out.release()
            cv2.destroyAllWindows()
        else:
            print('This function only works with opencv compatible movie files')

   