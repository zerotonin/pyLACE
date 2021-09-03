import pandas as pd
import numpy as np
import os, fishRecAnalysis
from counterCurrentAna import sortMultiFileFolder
from tqdm import tqdm

class fishDataBase():

    def __init__(self,dataBasePath = '/media/gwdg-backup/BackUp/Zebrafish/pythonDatabase',dbPos = None):
        self.dbPath   = dataBasePath
        if dbPos == None:
            self.dbPos = os.path.join(self.dbPath,'fishDataBase.csv')
        else:
            self.dbPos = dbPos
        self.loadDataBase()

    def loadDataBase(self):
        try:
            self.dataBase = pd.read_csv(self.dbPos)
            del self.dataBase['Unnamed: 0']
        except:
            answer ='?'
            while answer != 'y' and answer !='n':
                print('Fish data base cannot be read at position: ' + str(self.dbPos))
                answer = input('Do you want to create a fish data base at '+ str(self.dbPos)+ '? (y)es or (n)o' )
            if answer == 'n':
                raise ValueError('Cannot read fish data base')
            else:
                self.initDataBase()
    
    def initDataBase(self):
        dataBaseFields = ['genotype', 'sex', 'animalNo', 'fps', 'traceLenFrame', 
                  'traceLenSec', 'inZoneFraction', 'inZoneDuration', 
                  'inZoneMedDiverg_Deg', 'path2_inZoneBendability', 
                  'path2_midLineUniform_mm', 'path2_midLineUniform_pix', 
                  'path2_head_mm', 'path2_tail_mm', 'path2_probDensity', 
                  'path2_smr', 'path2_s2r', 'path2_seq', 'path2_csv', 
                  'path2_mat', 'path2_anaMat']
        self.dataBase = pd.DataFrame([],columns=dataBaseFields)
        self.dataBase.to_csv(self.dbPos)
        self.saveDataBase()
    
    def runMultiTraceFolder(self,folderPos,genName,expString):
        mff = sortMultiFileFolder(folderPos) 
        fileDict = mff.__main__() 
        for key in tqdm(fileDict.keys(),desc='analyse files'):
            dataDict = fileDict[key]
            fRAobj= fishRecAnalysis.fishRecAnalysis(dataDict,genName,expString)
            fRAobj.correctionAnalysis()
            dbEntry = fRAobj.saveDataFrames()
            self.addDataBase(dbEntry)
            self.saveDataBase()


    def addDataBase(self,dbEntry):
        self.dataBase  = self.dataBase.append(dbEntry,ignore_index=True)
    def saveDataBase(self):
        print(self.dbPos)
        self.dataBase.to_csv(self.dbPos)