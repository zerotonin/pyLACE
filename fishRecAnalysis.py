
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

    def saveResults(self):
        for dataListEntry in self.dataList:
            if dataListEntry[2] == 2:
                self.save2DMatrix(dataListEntry)
            elif dataListEntry[2] == 3:
                self.dataListEntry


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
        self.save2DMatrix(temp)
    
    def load2DMatrix(self,filePosition):
        return np.loadtxt(filePosition)
    
    def load3DMatrix(self,filePosition):
        temp = self.load2DMatrix(filePosition)
        return temp.reshape(temp.shape[0], temp.shape[1] // arr.shape[2], arr.shape[2])

