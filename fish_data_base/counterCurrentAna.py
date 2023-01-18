from pathlib import Path
import os, re

class sortMultiFileFolder:
    def __init__(self, source_path: Path, experiment_string: str):
        """
        Initialize the sortMultiFileFolder class.

        Args:
            source_path (Path): The path to the source folder where the files are located.
            experiment_string (str): The string used to identify the relevant files in the source folder.
        """
        self.source_path = source_path
        self.file_dict = dict()
        self.experiment_string = experiment_string

    def extract_genotype_number_sex(self, file_name: str, tag: str):
        """
        Extract the genotype, number, and sex information from a file name.

        Args:
            file_name (str): The name of the file to extract information from.
            tag (str): The string that precedes the genotype, number, and sex information in the file name.

        Returns:
            Tuple[str, int, str]: The genotype, number, and sex information extracted from the file name.
        """
        file_name = tag
        index = file_name.find(tag)
        genotype = file_name[index:index+2]
        sex = file_name[index+2:index+3]
        number = file_name[index+3:index+6]
        number = re.sub("[^0-9]", "", number)
        number = int(number)
        return genotype,number,sex

    def extract_genotype_number_sex_4intWT(self, file_name: str, tag: str):
        """
        Extract the genotype, number, and sex information from a file name for internal wild type fish.        
        
        Args:
            file_name (str): The name of the file to extract information from.   
            tag (str): The string that precedes the genotype, number, and sex information in the file name.

        Returns:
            Tuple[str, int, str]: The genotype, number, and sex information extracted from the file name.
        """
        index = file_name.find(tag)
        genotype = file_name[index:index+3]
        sex = file_name[index+3:index+4]
        number = file_name[index+4:index+7]
        #print(number,index,string)
        number = re.sub("[^0-9]", "", number)
        number = int(number)
        return genotype,number,sex

    def get_file_type(self, extension: str) -> str:
        '''
        getFileType - Returns the file type in lowercase format
        
        @extension - the file extension to be processed (string)
        
        Returns - the file type in lowercase format (string)
        '''
        return(extension[1::].lower())

    def make_dataset_key(self, genotype: str, animal_no: int, sex: str) -> str:
        """makeDataSetKey - Creates a unique key for a dataset

        Args:
            genotype (str): the genotype of the animal
            animal_no (int): the animal number
            sex (str): the sex of the animal

        Returns:
            str: the unique dataset key in the format: genotype+sex+animal_no (string)
        """
        return genotype + sex + str(animal_no)

    def classify_file(self,file_name,ext):
        """
        Given a file name and extension, classify the file into genotype, animal number, sex and file type

        Args:
        file_name (str): The file name
        ext (str): The file extension
        
        Returns:
        tuple: Tuple containing genotype, animal number, sex and file type
        """
        # convert file name to upper case
        fileNameUpper = file_name.upper()

        # Check if file name contains specific prefixes
        if 'HMF' in fileNameUpper:
            genotype,animalNo,sex =self.extract_genotype_number_sex(fileNameUpper,'HMF')
        elif 'HMM' in fileNameUpper:
            genotype,animalNo,sex =self.extract_genotype_number_sex(fileNameUpper,'HMM')
        elif 'HTF' in fileNameUpper:
            genotype,animalNo,sex =self.extract_genotype_number_sex(fileNameUpper,'HTF')
        elif 'HTM' in fileNameUpper:
            genotype,animalNo,sex =self.extract_genotype_number_sex(fileNameUpper,'HTM')
        elif 'INTF' in fileNameUpper:
            genotype,animalNo,sex =self.extract_genotype_number_sex_4intWT(fileNameUpper,'INTF')
        elif 'INTM' in fileNameUpper:
            genotype,animalNo,sex =self.extract_genotype_number_sex_4intWT(fileNameUpper,'INTM')
        elif 'INTWF' in fileNameUpper:
            fileNameUpper = fileNameUpper.replace('INTW','INT')
            genotype,animalNo,sex =self.extract_genotype_number_sex_4intWT(fileNameUpper,'INTF')
        elif 'INTWM'in fileNameUpper:
            fileNameUpper = fileNameUpper.replace('INTW','INT')
            genotype,animalNo,sex =self.extract_genotype_number_sex_4intWT(fileNameUpper,'INTM')
        else:
            genotype,animalNo,sex = ['N/A',-1,'N/A']
            print('file seems wrongly named: ',file_name)

        fileType = self.get_file_type(ext)

        return (genotype,animalNo,sex,fileType)    

    def update_file_dict(self,file_data_tuple,data_set_key,file_path):
        """
        This function updates the file dictionary with the file data tuple and the file path.
        If the data set key is not in the file dictionary, a new data dictionary is initialised.
        The file position is then updated in the data dictionary.
        
        Args
        ----
        file_data_tuple: tuple
            A tuple containing the file data.
        data_set_key: str
            The key for the data set.
        file_path: str
            The file path.
            
        Returns:
        --------
        None
        """
        # make new data set entry if data set is not in file dict
        if data_set_key not in self.file_dict.keys():
            dataDict = self.initialise_data_dict(file_data_tuple)
            self.file_dict[data_set_key] = dataDict
        # update file position in data dict
        self.update_data_dict(data_set_key,file_data_tuple,file_path)

    def initialise_data_dict(self,file_data_tuple): 
        """
        initialise_data_dict(file_data_tuple)

        This function takes in a tuple of file data and initializes a dictionary with keys and values from the tuple. It also calls a method 'get_full_experiment_name' and adds the result to the dictionary.

        Args:
        file_data_tuple (tuple): Tuple containing the genotype, animal number, and sex of the animal.

        Returns:
        dict: A dictionary containing the genotype, animal number, sex, experiment type, and empty values for smr, s2r, seq, csv, mat, and anaMat.

        """
        data_dict = dict()
        # add genotype key and value to dictionary
        data_dict['genotype'] = file_data_tuple[0]
        # add sex key and value to dictionary
        data_dict['sex'] = file_data_tuple[2]
        # add animalNo key and value to dictionary
        data_dict['animalNo'] = file_data_tuple[1]
        # add experiment name key and value to dictionary
        data_dict['expType'] = self.get_full_experiment_name()
        # add smr key and empty value to dictionary, this is the setup file from the Mauthner cell recordings
        data_dict['smr'] = ''
        # add s2r key and empty value to dictionary, this is the data ile from the Mauthner cell recordings
        data_dict['s2r'] = ''
        # add seq key and empty value to dictionary, this is the norpix movie file
        data_dict['seq'] = ''
        # add csv key and empty value to dictionary, this is the tank bounding box for specific files in which this is missing
        data_dict['csv'] = ''
        # add mat key and empty value to dictionary, this is the lace tracing file
        data_dict['mat'] = ''
        # add anaMat key and empty value to dictionary, this is the lace analysis file
        data_dict['anaMat'] = ''
        return data_dict
   
    
    def get_full_experiment_name(self):
        """
        get_full_experiment_name()

        This function returns the full name of the experiment based on the experiment_string attribute.

        Returns:
        str: The full name of the experiment. If the experiment_string is 'CCur', it will return 'counter current'. If the experiment_string is 'Ta', it will return 'motivated swimming'. If the experiment_string is 'Unt', it will return 'free swimming'. If the experiment_string is 'cst', it will return 'c-start'. If the experiment_string is not any of these values, it will raise a ValueError.

        """
        if self.experiment_string == 'CCur':
            return 'counter current'
        elif self.experiment_string == 'Ta':
            return 'motivated swimming'
        elif self.experiment_string == 'Unt':
            return 'free swiming'
        elif self.experiment_string == 'cst':
            return 'c-start'
        else:
            raise ValueError(f'sortMultiFileFolder: get_full_experiment_name: unknown experiment sting: f{self.experiment_string}')

    def update_data_dict(self,data_set_key,file_data_tuple,file_path):
        """
        update_data_dict(data_set_key, file_data_tuple, file_path)

        This function updates a file dictionary with the file path for a specific dataset.

        Args:
        data_set_key (str): the key of the dataset to update in the file dictionary
        file_data_tuple (tuple): tuple containing the genotype, animal number, sex, and data type of the file
        file_path (str): the path of the file

        """
        # with matlab files there can always be results_ana.mat and results.mat
        if file_data_tuple[3] == 'mat':
            if str(file_path)[-7:-4].lower() == 'ana':
                self.file_dict[data_set_key]['anaMat'] = str(file_path)
            else:
                self.file_dict[data_set_key]['mat'] = str(file_path)
        # all other dataset are individually
        else:
            self.file_dict[data_set_key][file_data_tuple[3]] = str(file_path)

    def __main__(self):

        """
        main()

        This function is the main method of the class, it will get all filenames in the source path, initializes a dictionary to store the files, and then for each file, it will classify the file, create a key for the dataset, and then update the file dictionary with the file path.

        Returns:
        dict: A dictionary containing the files classified by genotype, animal number, sex, and data type. The keys of the dictionary are a combination of genotype, animal number, and sex.

        """
        # get all filenames
        result = list(Path(self.source_path).rglob("*.*"))
        self.file_dict = dict()

        for file_path in result:
            file_name,ext  = os.path.splitext(os.path.basename(file_path))
            file_data_tuple = self.classify_file(file_name,ext)
            data_set_key    = self.make_dataset_key(file_data_tuple[0],file_data_tuple[1],file_data_tuple[2])
            self.update_file_dict(file_data_tuple,data_set_key,file_path)
        return self.file_dict


