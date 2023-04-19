#import cv2
from traceCorrector import traceCorrector
import traceAnalyser
import fishPlot
import pandas as pd
import numpy as np
import yaml
import os
from copy import deepcopy

class fishRecAnalysis():
    """
    A class used to analyze and process zebrafish recordings in a pandas DataFrame.

    ...

    Attributes
    ----------
    dbPath : str
        Path to the database folder containing the data files.
    dataDict : dict
        Dictionary containing the metadata of the recording.
    genName : str
        Name of the genotype.
    expStr : str
        Experiment string describing the type of experiment.
    arena_sizes : dict
        Dictionary containing the known arena sizes for different experiment types.

    Methods
    -------
    makeSaveFolder()
        Create a folder to save the analysis results.

    correctionAnalysis()
        Perform trace correction and analysis on the data.

    prepDf_3D(col1Name, col2Name, reps)
        Prepare a 3D DataFrame with given column names and repetitions.

    getTimeIndex(dataDF)
        Get the time index for the DataFrame.

    makePandasDF_3D(data, col1Name, col2Name, index=None)
        Create a 3D pandas DataFrame.

    makePandasDF_2D(data, col1Name, col2Name, index=None)
        Create a 2D pandas DataFrame.

    makePandasDF4Hist(data)
        Create a pandas DataFrame for histogram data.

    makeResultDFs()
        Create a dictionary containing all result DataFrames.

    saveDataFrames()
        Save all result DataFrames as CSV files.

    makeDataBaseEntry()
        Create a database entry with relevant information.

    save2DMatrix(dataListEntry)
        Save a 2D matrix to a file.

    save3DMatrix(dataListEntry)
        Save a 3D matrix to a file.

    load2DMatrix(filePosition)
        Load a 2D matrix from a file.

    load3DMatrix(filePosition)
        Load a 3D matrix from a file.

    check_mm_trace(default_answer='x')
        Check if the mm trace coordinates are within the expected arena dimensions.

    wrong_arena_dlg(expected_size, default_answer='x')
        Display a message about wrong arena dimensions and prompt for user input.

    interp_trace_mm(y_length, x_length, y_old, x_old)
        Interpolate the trace data based on corrected arena dimensions.
    """
    
    def __init__(self,dataDict,genName,expStr,birthDate,dataBasePath = '/media/gwdg-backup/BackUp/Zebrafish/pythonDatabase'):
        self.dbPath   = dataBasePath
        self.dataDict = dataDict
        self.genName  = genName
        self.dataDict['genotype'] = self.genName+'-'+self.dataDict['genotype']
        self.dataDict['birthDate'] = birthDate
        self.expStr   = expStr

        # self known arena sizes experiment type key tuple is (y,x)
        self.arena_sizes = {'cruise':(114,248),'c_start':(40,80),'counter_current':(45,167)}
    
    def makeSaveFolder(self):
        recNumber = len([os.path.join(self.dbPath, o) for o in os.listdir(self.dbPath)  if os.path.isdir(os.path.join(self.dbPath,o))])
        folderName = '{}_{}_{}_{}_{}_ID#{}'.format(self.expStr,self.dataDict['genotype'],self.dataDict['birthDate'],self.dataDict['sex'],self.dataDict['animalNo'],recNumber)
        self.savePath = os.path.join(self.dbPath,folderName)
        os.mkdir(self.savePath)
        #return folderName

    def correctionAnalysis(self):
        self.traCor = traceCorrector(self.dataDict)
        # calibrate the movie if nescessary
        if self.traCor.mmTraceAvailable == False:
            self.traCor.calibrateTracking()
        self.traCor.close_figure()
        # do pixel to mm conversion if nescessary
        self.traAna = traceAnalyser.traceAnalyser(self.traCor)
        if self.traCor.mmTraceAvailable == False:
            self.traAna.pixelTrajectories2mmTrajectories()
        # check if coordinates are in arena
        self.check_mm_trace(default_answer='x')
        # analysis depending on experiment type
        if self.expStr == 'CCur':
            self.traAna.calculateSpatialHistogram()
            self.traAna.inZoneAnalyse()
        self.traAna.getUniformMidLine()
        self.traAna.exportMetaDict()
        self.dataList = self.traAna.exportDataList()


    def prepDf_3D(self,col1Name,col2Name,reps):
        columnLabels = np.linspace(0,reps-1,reps,dtype=int)
        columnLabels = [[col1Name+'_'+str(x),col2Name+'_'+str(x)] for x in columnLabels]
        columnLabels = [j for sub in columnLabels for j in sub]
        return pd.DataFrame([],columns=columnLabels),columnLabels

    def getTimeIndex(self,dataDF):
        dataDF.index= dataDF.index/self.traAna.fps
        dataDF.index.name = 'time sec'
        return dataDF #np.linspace(0,self.traAna.traceLenSec,self.traAna.traceLenFrame)

    def makePandasDF_3D(self,data,col1Name,col2Name,index=None):
        reps = max([len(entry) for entry in data])
        dataDF,colLabels = self.prepDf_3D(col1Name,col2Name,reps)
        for  detection in data:
            entryDict = dict(zip(colLabels, detection.flatten()))
            dataDF    = dataDF.append(entryDict,ignore_index=True)
        
        if index == 'Time':
            dataDF = self.getTimeIndex(dataDF)

        return dataDF

    def makePandasDF_2D(self,data,col1Name,col2Name,index=None):
        dataDF = pd.DataFrame(data,columns= [col1Name,col2Name])
        if index == 'Time':
            dataDF = self.getTimeIndex(dataDF)
        return dataDF
       
    def makePandasDF4Hist(self,data):
   
        streamAxis = np.linspace(0,162,data.shape[1]+2)
        orthoAxis  = np.linspace(0,43,data.shape[0]+2)
        columnLabels = ['streamAxisMM ' +str(int(x)) for x in streamAxis[1:-1]]
        columnLabels = ['orthoIndexMM']+ columnLabels
        data = np.vstack((orthoAxis[1:-1],data.T)).T
        dataDF =pd.DataFrame(data,columns=columnLabels)
        dataDF.set_index('orthoIndexMM')
        return dataDF


    def makeResultDFs(self):
        returnDict = {'inZoneBendability': None,'midLineUniform_mm': None,
                'midLineUniform_pix': None,'head_mm': None,
                'tail_mm': None,'probDensity': None,'trace_mm':None}
        for data in self.dataList:

            if data[0] == 'inZoneBendability':
                returnDict['inZoneBendability'] = self.makePandasDF_3D(data[1],'bodyAxis','angle')
            elif data[0] =='midLineUniform_mm': 
                returnDict['midLineUniform_mm'] = self.makePandasDF_3D(data[1],'x_coord','y_coord','Time')
            elif data[0] =='midLineUniform_pix': 
                returnDict['midLineUniform_pix'] = self.makePandasDF_3D(data[1],'x_coord','y_coord','Time')
            elif data[0] =='head_mm': 
                returnDict['head_mm'] = self.makePandasDF_2D(data[1],'x_coord','y_coord','Time')
            elif data[0] =='tail_mm': 
                returnDict['tail_mm'] = self.makePandasDF_2D(data[1],'x_coord','y_coord','Time')
            elif data[0] =='trace_mm': 
                returnDict['trace_mm'] = pd.DataFrame(self.traAna.trace_mm,columns=['x_position_mm','y_position_mm','yaw_rad','thrust_m/s','slip_m/s','yaw_deg/s'])
            elif data[0] =='probDensity': 
                returnDict['probDensity']   = self.makePandasDF4Hist(data[1])
        
        return returnDict

    def saveDataFrames(self):
        dataFrames = self.makeResultDFs()
        self.makeSaveFolder()
        for key in dataFrames.keys():
            if not isinstance(dataFrames[key],type(None)):
                savePos = os.path.join(self.savePath,key+'.csv')
                self.dataDict['path2_'+key] = savePos
                dataFrames[key].to_csv(savePos)
        dbEntry = self.makeDataBaseEntry()
        return dbEntry

    def makeDataBaseEntry(self):
        dbEntry = self.dataDict.copy()
        del dbEntry['movieFrameIDX']
        del dbEntry['probDensity_xCenters']
        del dbEntry['probDensity_yCenters']
        dbEntry['path2_smr'] = dbEntry['smr']
        dbEntry['path2_s2r'] = dbEntry['s2r']
        dbEntry['path2_seq'] = dbEntry['seq']
        dbEntry['path2_csv'] = dbEntry['csv']
        dbEntry['path2_mat'] = dbEntry['mat']
        dbEntry['path2_anaMat'] = dbEntry['anaMat']
        del dbEntry['smr']
        del dbEntry['s2r']
        del dbEntry['seq']
        del dbEntry['csv']
        del dbEntry['mat']
        del dbEntry['anaMat']
        return dbEntry

    def save2DMatrix(self,dataListEntry):

        tag = dataListEntry[0]
        mat = dataListEntry[1]

        fileName = tag + '.txt'
        filePosition = os.path.join(self.savePath,fileName)

        np.savetxt(filePosition,mat)
        self.dataDict['path2_'+tag] = filePosition
    
    def save3DMatrix(self,dataListEntry):
        temp = deepcopy(dataListEntry)
        temp[1] = temp[1].reshape(temp[1].shape[0], -1)
        print(temp.shape)
        self.save2DMatrix(temp)
    
    def load2DMatrix(self,filePosition):
        return np.loadtxt(filePosition)
    
    def load3DMatrix(self,filePosition):
        temp = self.load2DMatrix(filePosition)
        return temp.reshape(temp.shape[0], temp.shape[1] // temp.shape[2], temp.shape[2])

    def check_mm_trace(self,default_answer='x'):
        if self.expStr == 'CCur':
            if np.max(self.traAna.trace_mm[:,0]) > self.arena_sizes['counter_current'][0] or np.max(self.traAna.trace_mm[:,1]) > self.arena_sizes['counter_current'][1]:
                self.wrong_arena_dlg(self.arena_sizes['counter_current'],default_answer)
        elif self.expStr == 'Ta' or self.expStr == 'Unt' :
            if np.max(self.traAna.trace_mm[:,0]) > self.arena_sizes['cruise'][0] or np.max(self.traAna.trace_mm[:,1]) > self.arena_sizes['cruise'][1]:
                self.wrong_arena_dlg(self.arena_sizes['cruise'],default_answer)
        elif self.expStr == 'cst':
            if np.max(self.traAna.trace_mm[:,0]) > self.arena_sizes['c_start'][0] or np.max(self.traAna.trace_mm[:,1]) > self.arena_sizes['c_start'][1]:
                self.wrong_arena_dlg(self.arena_sizes['c_start'],default_answer)
        else:
            raise ValueError('fishRecAnalysis: check_mm_trace: Unknown experiment type: ' + str(self.expStr))

    
    def wrong_arena_dlg(self,expected_size,default_answer='x'):
        print('===============================================================================')
        print('| The current file has trajectory coordinates outside the experimental setup! |')
        print('===============================================================================')
        print(' ')
        print('analysed MatLab file: ', self.dataDict['anaMat'])
        print('experiment string: ', self.expStr, ' | expected arena size (y,x): ', expected_size)
        print('found maximal coordinates (y,x):', (np.max(self.traAna.trace_mm[:,0]),np.max(self.traAna.trace_mm[:,1])))

        while default_answer not in 'ACTSN':
            default_answer = input('Which arena was WRONGLY used? (A)bort, (C)ounter current, cruise (T)ank, C-(S)tart, or (N)one all is fine: ')
            default_answer = default_answer.upper()

        if default_answer == 'A':
            raise ValueError('Aborted file due to user input: ' + self.dataDict['anaMat'])
        elif default_answer == 'C':
            self.interp_trace_mm(expected_size[0],expected_size[1],self.arena_sizes['counter_current'][0],self.arena_sizes['counter_current'][1])
        elif default_answer == 'T':
            self.interp_trace_mm(expected_size[0],expected_size[1],self.arena_sizes['cruise'][0],self.arena_sizes['cruise'][1])
        elif default_answer == 'S':
            self.interp_trace_mm(expected_size[0],expected_size[1],self.arena_sizes['c_start'][0],self.arena_sizes['c_start'][1])
        else:
            print('Nothing was changed')

        pass
    
    def interp_trace_mm(self,y_length,x_length,y_old,x_old):
        """ If the user entered the wrong dimensions of the arena and therefore
        wrongly calculated the mm trace this function can fix this. This

        WARNING the translational velocities will be approximations

        :param y_length: real y length of the arena
        :type y_length: float
        :param x_length: real x length of the arena
        :type x_length: float
        :param y_old: false y length of the arena
        :type y_old: float
        :param x_old: false x length of the arena
        :type x_old: float
        """
        y_factor = y_length/y_old
        x_factor = x_length/x_old
        mix_factor = (x_factor+y_factor)/2.0

        self.traAna.trace_mm[:,0]=self.traAna.trace_mm[:,0]*x_factor # matlab traces have x-coordinate in first position
        self.traAna.trace_mm[:,1]=self.traAna.trace_mm[:,1]*y_factor # matlab traces have y-coordinate in second position
        self.traAna.trace_mm[:,3]=self.traAna.trace_mm[:,3]*mix_factor
        self.traAna.trace_mm[:,4]=self.traAna.trace_mm[:,4]*mix_factor

        self.traAna.medMaxVelocities[:,0:2] = self.traAna.medMaxVelocities[:,0:2]*mix_factor
