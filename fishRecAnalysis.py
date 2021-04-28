
from traceCorrector import traceCorrector
import traceAnalyser
import fishPlot
import pandas as pd
import numpy as np
import yaml
import os
from copy import deepcopy

class fishRecAnalysis():
    
    def __init__(self,dataDict,genName,expStr,dataBasePath = '/media/gwdg-backup/BackUp/Zebrafish/pythonDatabase'):
        self.dataDict = dataDict
        self.dbPath   = dataBasePath
        self.genName  = genName
        self.expStr   = expStr
    
    def makeSaveFolder(self):
        recNumber = len([os.path.join(self.dbPath, o) for o in os.listdir(self.dbPath)  if os.path.isdir(os.path.join(self.dbPath,o))])
        folderName = '{}_{}_{}_{}_{}_ID#{}'.format(self.expStr,self.genName,self.dataDict['genotype'],self.dataDict['sex'],self.dataDict['animalNo'],recNumber)
        self.savePath = os.path.join(self.dbPath,folderName)
        os.mkdir(self.savePath)

    def correctionAnalysis(self):
        self.traCor = traceCorrector(self.dataDict)
        self.traCor.calibrateTracking()
        self.traAna = traceAnalyser.traceAnalyser(self.traCor)
        self.traAna.pixelTrajectories2mmTrajectories()
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

    def makePandasDF_3D(self,data,col1Name,col2Name,index=None):
        reps = max([len(entry) for entry in data])
        dataDF,colLabels = self.prepDf_3D(col1Name,col2Name,reps)
        for  detection in data:
            entryDict = dict(zip(colLabels, detection.flatten()))
            dataDF    = dataDF.append(entryDict,ignore_index=True)
        
        if index == 'Time':
            dataDF.index= dataDF.index/self.traAna.fps
            dataDF.index.name = 'time sec'

        return dataDF

    def saveResults(self):
        for dataListEntry in self.dataList:
            if dataListEntry[2] == 2:
                self.save2DMatrix(dataListEntry)
            elif dataListEntry[2] == 3:
                self.save3DMatrix(dataListEntry)


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
        return temp.reshape(temp.shape[0], temp.shape[1] // arr.shape[2], arr.shape[2])

