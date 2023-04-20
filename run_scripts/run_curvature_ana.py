from importlib import reload
from sre_compile import isstring
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fish_data_base.fishDataBase as fishDataBase
import seaborn as sns
from tqdm import tqdm
import trace_analysis.speed_analyser as speed_analyser
from scipy.signal import find_peaks

#%%
#multiFileFolder = '/media/gwdg-backup/BackUp/Vranda/data_counter_c-start/countercurrent_onefolder/rei_last_generation_11-2018'
#multiFileFolder = '/media/gwdg-backup/BackUp/Vranda/Finaldata_rei/Countercurrent_trials_rei'
#db = fishDataBase.fishDataBase("/home/bgeurten/fishDataBase/")
# Experiment types CCur counter current , Ta tapped, Unt untapped, cst, c-startz
#db.runMultiTraceFolder(multiFileFolder,'rei','CCur','11-2018',start_at=0)
#%%
db = fishDataBase.fishDataBase("/home/bgeurten/fishDataBase",'/home/bgeurten/fishDataBase/fishDataBase_cruise.csv')
db.rebase_paths()
df = db.database
curv_list = list()
for i,row in tqdm(df.iterrows()):
    if isinstance(row.path2_midLineUniform_pix,str):
        midline_df = pd.read_csv(row.path2_midLineUniform_pix)
        curv_list.append(get_total_curvature_amps(midline_df))
curv_df = pd.concat([df[['genotype', 'sex', 'animalNo', 'expType', 'birthDate']],pd.DataFrame(curv_list)],axis=1)
curv_df.to_csv("/home/bgeurten/PyProjects/reRandomStats/Data/rei_curvature_data.csv", index=False)







def calculate_total_curvature(df, number_of_coordinates=10):
    curvatures = []
    for index, row in df.iterrows():
        points = [(row[f'x_coord_{i}'], row[f'y_coord_{i}']) for i in range(number_of_coordinates)]
        tangents = np.diff(points, axis=0)
        normalized_tangents = tangents / np.linalg.norm(tangents, axis=1, keepdims=True)
        diff_normalized_tangents = np.diff(normalized_tangents, axis=0)
        magnitudes = np.linalg.norm(diff_normalized_tangents, axis=1)
        total_curvature = np.sum(magnitudes)
        curvatures.append(total_curvature)
    return np.array(curvatures)

def find_peak_amplitudes(curvature_vector, prominence_threshold):
    """
    Find peaks in the curvature vector and return their amplitudes.

    Parameters
    ----------
    curvature_vector : np.array
        The curvature vector containing the curvature values.
    prominence_threshold : float
        The prominence threshold to consider a peak as significant.

    Returns
    -------
    peak_amplitudes : np.array
        The amplitudes of the detected peaks.
    """
    peaks, _ = find_peaks(curvature_vector, prominence=prominence_threshold)
    peak_amplitudes = curvature_vector[peaks]
    return peak_amplitudes

def get_total_curvature_amps(midline_df,prominence_threshold = 0.5):

    total_curv = calculate_total_curvature(midline_df)
    total_curv_amps=find_peak_amplitudes(total_curv,prominence_threshold)
    median_curv_amp = np.median(total_curv_amps)
    mean_curv_amp = np.mean(total_curv_amps)
    max_curv_amp = np.max(total_curv_amps)

    return {'median_curv_amp':median_curv_amp,'mean_curv_amp':mean_curv_amp,'max_curv_amp':max_curv_amp}



import seaborn as sns
for expType in [('Ta','motivated swimming'),('Unt','free swimming')]:
    for parameter in ['median_curv_amp', 'mean_curv_amp', 'max_curv_amp']:
        f= plt.figure()
        sns.boxplot(x="genotype", y=parameter, order=['rei-INT', 'rei-HT', 'rei-HM'],
                hue="sex",hue_order=['M','F'],data=curv_df.loc[curv_df['expType']==expType[1],:]).set_title(expType[1])
        #plt.savefig('/home/bgeurten/fishDataBase/figures/'+f'{expType[1]}--{parameter}.svg'.replace(' ','_').replace('/','_per_'))
plt.show()
