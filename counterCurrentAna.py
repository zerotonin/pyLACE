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
        elif 'INTWF' in fileNameUpper:
            fileNameUpper = fileNameUpper.replace('INTW','INT')
            genotype,animalNo,sex =self.extractGenotypeNumberSex4intWT(fileNameUpper,'INTF')
        elif 'INTWM'in fileNameUpper:
            fileNameUpper = fileNameUpper.replace('INTW','INT')
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


