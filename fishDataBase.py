import pandas as pd
import numpy as np
import os, fishRecAnalysis
from counterCurrentAna import sortMultiFileFolder
from tqdm import tqdm

class fishDataBase():

    def __init__(self,data_base_path = '/media/gwdg-backup/BackUp/Zebrafish/pythonDatabase',data_base_file_position = None):
        self.database_path  = data_base_path
        if data_base_file_position == None:
            self.data_base_file_position = os.path.join(self.database_path,'fishDataBase.csv')
        else:
            self.data_base_file_position = data_base_file_position
        self.load_database()

    def load_database(self):
        try:
            self.database = pd.read_csv(self.data_base_file_position)
            del self.database['Unnamed: 0']
        except:
            answer ='?'
            while answer != 'y' and answer !='n':
                print('Fish data base cannot be read at position: ' + str(self.data_base_file_position))
                answer = input('Do you want to create a fish data base at '+ str(self.data_base_file_position)+ '? (y)es or (n)o' )
            if answer == 'n':
                raise ValueError('Cannot read fish data base')
            else:
                self.init_database()
    
    def init_database(self):
        data_base_fields = ['genotype', 'sex', 'animalNo','expType', 'birthDate','fps', 'traceLenFrame', 
                  'traceLenSec', 'inZoneFraction', 'inZoneDuration', 
                  'inZoneMedDiverg_Deg', 'path2_inZoneBendability', 
                  'path2_midLineUniform_mm', 'path2_midLineUniform_pix', 
                  'path2_head_mm', 'path2_tail_mm', 'path2_probDensity', 
                  'path2_smr', 'path2_s2r', 'path2_seq', 'path2_csv', 
                  'path2_mat', 'path2_anaMat']
        self.database = pd.DataFrame([],columns=data_base_fields)
        self.database.to_csv(self.data_base_file_position)
        self.saveDataBase()
    
    def runMultiTraceFolder(self,folder_position,gene_name,experiment_str,birth_date,start_at=0):
        mff = sortMultiFileFolder(folder_position,experiment_str) 
        file_dictionary = mff.__main__()
        keys = [k for k in file_dictionary.keys()] 
        allready_analysed_filenames = [os.path.basename(x) for x in self.database.path2_anaMat]

        for key in tqdm(keys[start_at::],desc='analyse files'):
            data_dictionary = file_dictionary[key]
            if os.path.basename(data_dictionary['anaMat']) not in allready_analysed_filenames:
                try:
                    fRAobj= fishRecAnalysis.fishRecAnalysis(data_dictionary,gene_name,experiment_str,birth_date)
                    fRAobj.correctionAnalysis()
                    database_entry = fRAobj.saveDataFrames()
                    self.addDataBase(database_entry)
                    self.saveDataBase()
                except:
                    print('The following directory could not be analysed: '+ data_dictionary['anaMat'])


    def addDataBase(self,database_entry):
        self.database  = self.database.append(database_entry,ignore_index=True)

    def integrateDataBase(self,db_to_integrate_fPos):
        # shorthand
        new_df = pd.read_csv(db_to_integrate_fPos)
        del new_df['Unnamed: 0']
        self.database  = self.database.append(new_df,ignore_index=True)
    
    def saveDataBase(self):
        print(self.data_base_file_position)
        self.database.to_csv(self.data_base_file_position)