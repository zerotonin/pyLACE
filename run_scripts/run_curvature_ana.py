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
#multiFileFolder = '/media/gwdg-backup/BackUp/Vranda/data_counter_c-start/countercurrent_onefolder/rei_last_generation_11-2018'
#multiFileFolder = '/media/gwdg-backup/BackUp/Vranda/Finaldata_rei/Countercurrent_trials_rei'
#db = fishDataBase.fishDataBase("/home/bgeurten/fishDataBase/")
# Experiment types CCur counter current , Ta tapped, Unt untapped, cst, c-startz
#db.runMultiTraceFolder(multiFileFolder,'rei','CCur','11-2018',start_at=0)
#%%
db = fishDataBase.fishDataBase("/home/bgeurten/fishDataBase",'/home/bgeurten/fishDataBase/fishDataBase_counter_current.csv')
db.rebase_paths()
df = db.database
curv_list = list()
for i,row in tqdm(df.iterrows()):
    if isinstance(row.path2_midLineUniform_pix,str):
        try:
            midline_df = pd.read_csv(row.path2_midLineUniform_pix)
            ca = CurvatureAnalyser.CurvatureAnalyser(midline_df)
            curv_list.append(ca.get_total_curvature_amps())
        except:
            print(f'!! {row.path2_midLineUniform_pix} did not produce output')
curv_df = pd.concat([df[['genotype', 'sex', 'animalNo', 'expType', 'birthDate']],pd.DataFrame(curv_list)],axis=1)
curv_df.to_csv("/home/bgeurten/PyProjects/reRandomStats/Data/rei_curvature_ccur_data.csv", index=False)


import seaborn as sns
for expType in [('Ta','motivated swimming'),('Unt','free swimming')]:
    for parameter in ['median_curv_amp', 'mean_curv_amp', 'max_curv_amp']:
        f= plt.figure()
        sns.boxplot(x="genotype", y=parameter, order=['rei-INT', 'rei-HT', 'rei-HM'],
                hue="sex",hue_order=['M','F'],data=curv_df.loc[curv_df['expType']==expType[1],:]).set_title(expType[1])
        #plt.savefig('/home/bgeurten/fishDataBase/figures/'+f'{expType[1]}--{parameter}.svg'.replace(' ','_').replace('/','_per_'))
plt.show()
