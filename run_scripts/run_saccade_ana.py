from importlib import reload
from sre_compile import isstring
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fish_data_base.fishDataBase as fishDataBase
import seaborn as sns
from tqdm import tqdm
import trace_analysis.CurvatureAnalyser as CurvatureAnalyser

#%%
db = fishDataBase.fishDataBase("/home/bgeurten/fishDataBase",'/home/bgeurten/fishDataBase/fishDataBase_cruise.csv')
db.load_database()
df = db.database
#%%
for row in df.iterrows():
    pass