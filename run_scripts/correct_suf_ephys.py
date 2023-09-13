import pandas as pd
import numpy as np
import fish_data_base.fishDataBase as fishDataBase
import re
from tqdm import tqdm

def extract_info_from_id_text(identifier_text):
    """
    Extract relevant information from an identifier text.
    
    Given an identifier text, this function extracts the strain type, sex, 
    and fish number based on pre-defined mappings and string patterns.
    
    Args:
        identifier_text (str): The identifier text to be parsed.
        
    Returns:
        tuple: A tuple containing the strain type (str), sex (str), and fish number (int).
    
    Example:
        >>> extract_info_from_id_text("sample_Hm123M4IIII")
        ('sufge1-HM', 'M', 4)
    """
    
    # Mapping of short strain identifiers to full identifiers
    strain_map = {'Ht':'sufge1-HT', 'Hm':'sufge1-HM', 'Int':'sufge1-INT'}
    
    # Extract the identifier part from the input string
    id = identifier_text.split('_')[1]
    
    # Determine the strain and the rest of the ID based on the pattern
    if id[1] == 'n':
        strain = id[:3]
        rest = id[3:]
    else:
        strain = id[:2]
        rest = id[2:]
        
    # Map the short strain identifier to the full identifier
    strain = strain_map[strain]
    
    # Extract the sex from the rest of the ID
    sex = rest[:1]
    rest = rest[1:]
    
    # Extract the fish number using regex to find digits
    fish_no = int(re.findall(r'\d+', rest)[0])
    
    return strain, sex, fish_no


def get_val(df,field):
    """
    Retrieve the first value in a specified DataFrame column.
    
    Args:
        df (pd.DataFrame): The DataFrame from which to retrieve the value.
        field (str): The column name.
        
    Returns:
        any: The first value in the specified column.
    """
    return  df[field].iloc[0]


def shift_dataframe(df, y, correct_timing = True):
    """
    Shifts the DataFrame rows in a cyclic manner.

    The DataFrame is rearranged such that it starts from the row indexed y+1, 
    followed by the rows indexed from 0 to y at the end.
    
    Args:
        df (pd.DataFrame): The DataFrame to be shifted.
        y (int): The index up to which rows will be shifted to the end.
        
    Returns:
        pd.DataFrame: The rearranged DataFrame.
    """

    # Rearrange the DataFrame
    upper_part = df.iloc[y+1:].copy()
    lower_part = df.iloc[:y+1].copy()

    if correct_timing:
        # Calculate the median frame duration
        frame_dur = df['time sec'].diff().median()

        # Correct the 'time sec' column
        upper_part.loc[:, 'time sec'] = upper_part['time sec'] - upper_part['time sec'].iloc[0]
        lower_part.loc[:, 'time sec'] = lower_part['time sec'] + upper_part['time sec'].iloc[-1] + frame_dur
    
    # Append the rearranged DataFrames
    new_df = upper_part.append(lower_part).reset_index(drop=True)
    
    return new_df




def correct_rr_error(fish_df,rr_offset):
        
    """
    Corrects tracking data errors in specified fields by shifting rows.
    
    Reads tracking data from a set of pre-defined fields in a DataFrame, shifts the rows by the given offset, 
    and potentially saves them back to the original file.
    
    Args:
        fish_df (pd.DataFrame): DataFrame containing the file paths to tracking data.
        rr_offset (int): Row offset by which to shift the data.
        
    """
    tracking_fields = ['path2_trace_mm','path2_midLineUniform_mm', 'path2_midLineUniform_pix', 'path2_head_mm', 'path2_tail_mm']
    
    for field in tracking_fields:
        path =get_val(fish_df,field)
        if type(path) == str:
            df = pd.read_csv(path)
            if field == 'path2_trace_mm':
                df = shift_dataframe(df,rr_offset,correct_timing=False)
            else:
                df = shift_dataframe(df,rr_offset,correct_timing=True)

            # Uncomment to save back to CSV
            #df.to_csv(path,index=False)


######################################################
###############  WARNING - READ THIS!  ###############
######################################################
#                                                    #
#  This script applies a one-time correction to the  #
#  data. If this correction has already been run     #
#  previously, running it again will CORRUPT the     #
#  data. Confirm that you have not already executed  #
#  this script on the data you're about to process.  #
#                                                    #
#                RAN ON 12th SEPT 2023               #
#                                                    #
######################################################
###############  WARNING - READ THIS!  ###############
######################################################
'''
db = fishDataBase.fishDataBase("/home/bgeurten/fishDataBase",'/home/bgeurten/fishDataBase/fishDataBase_cstart.csv')
#db.rebase_paths()
df = db.database
df_jump = pd.read_csv('/home/bgeurten/fishDataBase/suf_cstart_round_robin_jumps.csv')

for i,row in tqdm(df_jump.iterrows(),desc= 'round robin update'):
    strain, sex, fish_no = extract_info_from_id_text(row[1])
    # get original fish data
    temp = df.loc[(df['genotype'] == strain) & (df['sex'] == sex) & (df['animalNo'] == fish_no), :]
    if len(temp) == 1:
        rr_offset = int(row[2].split('-')[0])
        correct_rr_error(temp, rr_offset)
'''
######################################################
###############  WARNING - READ THIS!  ###############
######################################################
#                                                    #
#  Remember: The data correction has now been        #
#  applied. Do not run this script again on the      #
#  same dataset.                                     #
#                                                    #
#                RAN ON 12th SEPT 2023               #
#                                                    #;
######################################################
###############  WARNING - READ THIS!  ###############
######################################################







