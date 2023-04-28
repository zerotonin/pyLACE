import pandas as pd

class EthoVisionReader:
    """
    A class to read EthoVision Excel files and extract trajectory and metadata.

    Attributes:
        filename (str): The filename of the Excel file to read.
        excel_data (dict): A dictionary containing the data from the Excel file.

    Author: B. Geurten
    Date: 28th April 2023
    """

    def __init__(self, filename):
        """
        Constructs the EthoVisionReader object with the given filename.

        Args:
            filename (str): The filename of the Excel file to read.
        """
        self.filename = filename
        self.excel_data = self.read_file()

    def read_file(self):
        """
        Reads the Excel file and stores the data in a dictionary.

        Returns:
            dict: A dictionary containing the data from the Excel file.
        """
        return pd.read_excel(self.filename, sheet_name=None)

    def get_trajectory(self, sheet_data):
        """
        Extracts the trajectory data from the given sheet data.

        Args:
            sheet_data (DataFrame): The data from a single sheet in the Excel file.

        Returns:
            DataFrame: A DataFrame containing the trajectory data.
        """
        column_heads = sheet_data.iloc[35,:].to_list()
        column_units = sheet_data.iloc[36,:].to_list()
        col_combi = list(zip(column_heads, column_units))
        column_names = [f'{x[0]}_{x[1]}'.replace(' ','_') for x in col_combi]

        return pd.DataFrame(sheet_data.iloc[37::,:].to_numpy(), columns=column_names)

    def get_meta_data(self, sheet_data, df):
        """
        Adds metadata to the given DataFrame.

        Args:
            sheet_data (DataFrame): The data from a single sheet in the Excel file.
            df (DataFrame): A DataFrame containing the trajectory data.

        Returns:
            DataFrame: A DataFrame containing the trajectory data with metadata.
        """
        meta_keys = ['Tank_number', 'Sex', 'ID', 'Start time', 'Arena ID', 'Trial ID', 'Subject ID']
        
        for key in meta_keys:
            value = sheet_data.loc[sheet_data.iloc[:, 0] == key].iloc[:, 1].values
            df[key.replace(' ', '_')] = value[0] if len(value) > 0 else None

        return df

    def main(self):
        """
        Main function to process the data from the Excel file.

        Returns:
            DataFrame: A DataFrame containing the concatenated data from all sheets.
        """
        df_list = list()

        for sheet_name, sheet_data in self.excel_data.items():
            if sheet_data.iloc[-1, 0] != 'No samples logged for this track!':
                df_trajectory = self.get_trajectory(sheet_data)
                df_meta_data = self.get_meta_data(sheet_data, df_trajectory)
                df_list.append(df_meta_data)

        final_data = pd.concat(df_list)
        return final_data

# Example usage:
# filename = "/home/bgeurten/Downloads/Raw_data-2023_setup-Trial1.xlsx"
# etho_vision_reader = EthoVisionReader(filename)
# final_data = etho_vision_reader.main()
# print(final_data)