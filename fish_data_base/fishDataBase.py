import pandas as pd
import os, fish_data_base.fishRecAnalysis as fishRecAnalysis
from fish_data_base.counterCurrentAna import sortMultiFileFolder
from tqdm import tqdm

class fishDataBase():
    """
    This class is used to create and manipulate a database of fish analysis data.
    """
    def __init__(self,database_path = '/media/gwdg-backup/BackUp/Zebrafish/pythonDatabase',database_file_position = None):
        """Initializes the FishDataBase class by specifying the path to the database and the position of the database file.

        Args:
        database_path (str): The path to the database folder.
        database_file_position (str): The position of the database file. If None, it will default to the database_path/fishDataBase.csv.
        """
        self.database_path  = database_path
        self.data_paths = ['path2_inZoneBendability',
                           'path2_midLineUniform_mm', 'path2_midLineUniform_pix', 'path2_head_mm',
                           'path2_tail_mm', 'path2_probDensity', 'path2_trace_mm', 'path2_probDensity', 'path2_smr',
                           'path2_csv']
        if database_file_position == None:
            self.database_file_position = os.path.join(self.database_path,'fishDataBase.csv')
        else:
            self.database_file_position = database_file_position
        self.load_database()


    def load_database(self):
        """
        Attempts to load the database from the specified file position. If it is unable to do so, it will ask the user if they want to create a new database.
        """
        try:
            self.database = pd.read_csv(self.database_file_position)
            try:
                del self.database['Unnamed: 0']
            except:
                pass
        except:
            answer ='?'
            while answer != 'y' and answer !='n':
                print('Fish data base cannot be read at position: ' + str(self.database_file_position))
                answer = input('Do you want to create a fish data base at '+ str(self.database_file_position)+ '? (y)es or (n)o' )
            if answer == 'n':
                raise ValueError('Cannot read fish data base')
            else:
                self.init_database()

    def init_database(self):
        """
        Initializes an empty database with the specified fields.
        """

        data_base_fields = ['genotype', 'sex', 'animalNo','expType', 'birthDate','fps', 'traceLenFrame', 
                  'traceLenSec', 'inZoneFraction', 'inZoneDuration', 
                  'inZoneMedDiverg_Deg', 'path2_inZoneBendability', 
                  'path2_midLineUniform_mm', 'path2_midLineUniform_pix', 
                  'path2_head_mm', 'path2_tail_mm', 'path2_probDensity', 
                  'path2_smr', 'path2_s2r', 'path2_seq', 'path2_csv', 
                  'path2_mat', 'path2_anaMat']
        self.database = pd.DataFrame([],columns=data_base_fields)
        self.database.to_csv(self.database_file_position)
        self.saveDataBase()
    
    def run_multi_trace_folder(self, folder_position, gene_name, experiment_str, birth_date, start_at=0, gui_correction=True,default_wrong_arena = ''):
        """
        Analyzes multiple trace files in a given folder and adds the data to the database.

        Args:
            folder_position (str): The path to the folder containing the trace files.
            gene_name (str): The gene name associated with the trace files.
            experiment_str (str): The name of the experiment associated with the trace files.
            birth_date (str): The birth date of the fish associated with the trace files.
            start_at (int, optional): The index at which to start analyzing the trace files. Defaults to 0.
            gui_correction (bool, optional): Whether to apply GUI-based correction during analysis. Defaults to True.
        """
        # Sort and organize the files in the folder
        mff = sortMultiFileFolder(folder_position, experiment_str)
        file_dictionary = mff.__main__()

        # Get a list of file keys
        keys = [k for k in file_dictionary.keys()]

        # Get a list of filenames that have already been analyzed
        allready_analysed_filenames = [os.path.basename(x) for x in self.database.path2_anaMat]

        # Iterate through the files and analyze them if they have not been analyzed yet
        for key in tqdm(keys[start_at::], desc='analyse files'):
            data_dictionary = file_dictionary[key]
            if os.path.basename(data_dictionary['anaMat']) not in allready_analysed_filenames:
                # Perform the analysis on the current file
                fRAobj = fishRecAnalysis.fishRecAnalysis(data_dictionary, gene_name, experiment_str, birth_date, self.database_path)
                fRAobj.main(correction_mode=gui_correction)

                # Add the analysis result to the database
                database_entry = fRAobj.saveDataFrames()
                self.addDataBase(database_entry)

                # Save the updated database
                self.saveDataBase()



    def addDataBase(self,database_entry):
        """
        Adds a new entry to the database.
        
        Args:
            database_entry (dict): The data to be added to the database.
        """
        self.database  = self.database.append(database_entry,ignore_index=True)

#    def integrate_database(self,db_to_integrate_fPos):
#        # shorthand
#        new_df = pd.read_csv(db_to_integrate_fPos)
#        del new_df['Unnamed: 0']
#        self.database  = self.database.append(new_df,ignore_index=True)
    def integrate_database(self, file_path): # new version untested
        """
        Integrates a new database into the current one.

        Args:
            file_path (str): The file path of the database to be integrated.
        """
        new_dataframe = pd.read_csv(file_path)
        new_dataframe.drop(columns=['Unnamed: 0'], inplace=True)
        self.database = self.database.append(new_dataframe, ignore_index=True)

    def rebase_paths(self, default_path='/media/gwdg-backup/BackUp/Zebrafish/pythonDatabase'):
        """
        Rebases all database entry paths to the position of the databases csv file.

        Args:
            default_path (str, optional): The current path to be replaced. Defaults to '/media/gwdg-backup/BackUp/Zebrafish/pythonDatabase'.
        """
        for path_column in self.data_paths:
            self.database[path_column] = self.database[path_column].replace({default_path: self.database_path}, regex=True)


    def saveDataBase(self):
        """
        Saves the current state of the database to the specified file position.
        """
        print(self.database_file_position)
        self.database.to_csv(self.database_file_position,index=False)