
import numpy as np
import pandas as pd
from pandas.core import api
from pandas.core.arrays.boolean import BooleanArray
import quantities as pq
import data_handlers.spike2SimpleIO as spike2SimpleIO 
import seaborn as sns
import matplotlib.pyplot as plt




fN = '/media/gwdg-backup/BackUp/Vranda/BielefeldExp/rei_Cstart/IntW_06-2020repeat/rei_cstIntWF2II_06-2020.smr'
s2sr = spike2SimpleIO.spike2SimpleReader(fN)
s2sr.main()
segSav = spike2SimpleIO.segmentSaver(s2sr,'/media/dataSSD/testPanda.h5')
df = segSav.main()
