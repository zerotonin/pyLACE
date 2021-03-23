from pathlib import Path
import os,re
import numpy as np
import scipy.io

class sortMultiFileFolder():

    def __init__(self,sourcePath):
        self.sourcePath = sourcePath
        self.fileDict   = dict()

    def extractGenotypeNumberSex(self,string,tag):
        index = string.find(tag)
        genotype = string[index:index+2]
        sex = string[index+2:index+3]
        number = string[index+3:index+6]
        number = re.sub("[^0-9]", "", number)
        number = int(number)
        return genotype,number,sex

    def extractGenotypeNumberSex4intWT(self,string,tag):
        index = string.find(tag)
        genotype = string[index:index+3]
        sex = string[index+3:index+4]
        number = string[index+4:index+7]
        #print(number,index,string)
        number = re.sub("[^0-9]", "", number)
        number = int(number)
        return genotype,number,sex

    def getFileType(self,extension):
        return(extension[1::].lower())

    def makeDataSetKey(self,genotype,animalNo,sex):
        return genotype+sex+str(animalNo)

    def classifyFile(self,fileName,ext):
        fileNameUpper = fileName.upper()
        if 'HMF' in fileNameUpper:
            genotype,animalNo,sex =self.extractGenotypeNumberSex(fileNameUpper,'HMF')
        elif 'HMM' in fileNameUpper:
            genotype,animalNo,sex =self.extractGenotypeNumberSex(fileNameUpper,'HMM')
        elif 'HTF' in fileNameUpper:
            genotype,animalNo,sex =self.extractGenotypeNumberSex(fileNameUpper,'HTF')
        elif 'HTM' in fileNameUpper:
            genotype,animalNo,sex =self.extractGenotypeNumberSex(fileNameUpper,'HTM')
        elif 'INTF' in fileNameUpper:
            genotype,animalNo,sex =self.extractGenotypeNumberSex4intWT(fileNameUpper,'INTF')
        elif 'INTM' in fileNameUpper:
            genotype,animalNo,sex =self.extractGenotypeNumberSex4intWT(fileNameUpper,'INTM')
        else:
            genotype,animalNo,sex = ['N/A',-1,'N/A']
            print('file seems wrongly named: ',fileName)

        fileType = self.getFileType(ext)

        return (genotype,animalNo,sex,fileType)    

    def updatefileDict(self,fileDataTuple,dataSetKey,filePath):
        # make new data set entry if data set is not in file dict
        if dataSetKey not in self.fileDict.keys():
            dataDict = self.initialiseDataDict(fileDataTuple)
            self.fileDict[dataSetKey] = dataDict
        # update file position in data dict
        self.updateDataDict(dataSetKey,fileDataTuple,filePath)

    def initialiseDataDict(self,fileDataTuple): 
        dataDict = dict()
        dataDict['genotype'] = fileDataTuple[0] 
        dataDict['sex']      = fileDataTuple[2]
        dataDict['animalNo'] = fileDataTuple[1]
        dataDict['smr']      = '' 
        dataDict['s2r']      = ''
        dataDict['seq']      = ''
        dataDict['csv']      = ''
        dataDict['mat']      = ''
        dataDict['anaMat']   = ''
        return dataDict       
    
    def updateDataDict(self,dataSetKey,fileDataTuple,filePath):
        # with matlab files there can always be results_ana.mat and results.mat
        if fileDataTuple[3] == 'mat':
            if str(filePath)[-7:-4].lower() == 'ana':
                self.fileDict[dataSetKey]['anaMat'] = str(filePath)
            else:
                self.fileDict[dataSetKey]['mat'] = str(filePath)
        # all other dataset are individually
        else:
            self.fileDict[dataSetKey][fileDataTuple[3]] = str(filePath)

    def __main__(self):
        # get all filenames
        result = list(Path(self.sourcePath).rglob("*.*"))
        self.fileDict = dict()

        for filePath in result:
            fileName,ext  = os.path.splitext(os.path.basename(filePath))
            fileDataTuple = self.classifyFile(fileName,ext)
            dataSetKey    = self.makeDataSetKey(fileDataTuple[0],fileDataTuple[1],fileDataTuple[2])
            self.updatefileDict(fileDataTuple,dataSetKey,filePath)
        return self.fileDict


class matLabResultLoader():

    def __init__(self, filePosition, mode ='anaMat'):
        self.filePosition = filePosition
        self.mode         = mode

    def readAnaMatFile(self): 
        # read matlab analysis
        mat = scipy.io.loadmat(self.filePosition)
        self.metaData     = mat['metaData']
        self.analysedData = mat['analysedData']
        self.traceResult  = self.analysedData[0][0][0]

    def ndArray2npArray2D(self,ndArray):
        temp = ndArray.tolist()
        return np.fliplr(np.array([x[0][:] for x in temp])) # fliplr as x should be first

    def flattenNDarray(self,ndArray):
        temp = ndArray.tolist()
        return np.array([np.fliplr(x[0][0]) for x in temp]) # fliplr as x should be first
    
    def splitResults2Variables(self):
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
        self.traceInfo        = self.ndArray2npArray2D(self.traceResult[:,0])
        self.traceContour     = self.flattenNDarray(self.traceResult[:,1])
        self.traceMidline     = self.flattenNDarray(self.traceResult[:,2])
        self.traceHead        = self.ndArray2npArray2D(self.traceResult[:,3])
        self.traceTail        = self.ndArray2npArray2D(self.traceResult[:,4])
        self.trace            = self.analysedData[0][0][1]
        self.bendability      = self.analysedData[0][0][2]
        self.binnedBend       = self.analysedData[0][0][3]
        self.saccs            = self.analysedData[0][0][4]
        self.trigAveSacc      = self.analysedData[0][0][5]
        self.medMaxVelocities = self.analysedData[0][0][6] 
    
    def getData(self):
        if self.mode == 'anaMat':
            self.readAnaMatFile()
            self.splitResults2Variables()
            return self.traceInfo, self.traceContour, self.traceMidline, self.traceHead, self.traceTail, self.trace, self.bendability, self.binnedBend, self.saccs, self.trigAveSacc, self.medMaxVelocities
        else:
            raise ValueError('Unknown mode for matLabResultLoader: '+str(self.mode))
