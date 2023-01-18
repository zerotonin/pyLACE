from importlib import reload
from sre_compile import isstring
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fish_data_base.fishDataBase as fishDataBase
import seaborn as sns
from tqdm import tqdm
import speed_analyser

#%%
multiFileFolder = '/media/gwdg-backup/BackUp/Vranda/data_counter_c-start/countercurrent_onefolder/rei_last_generation_11-2018'
multiFileFolder = '/media/gwdg-backup/BackUp/Vranda/Finaldata_rei/Countercurrent_trials_rei'
db = fishDataBase.fishDataBase("/home/bgeurten/fishDataBase")
# Experiment types CCur counter current , Ta tapped, Unt untapped, cst, c-startz
#db.runMultiTraceFolder(multiFileFolder,'rei','CCur','11-2018',start_at=0)
#%%
db = fishDataBase.fishDataBase("/home/bgeurten/fishDataBase",'/home/bgeurten/fishDataBase/fishDataBase_cruise.csv')
db.rebase_paths()

sa = speed_analyser.speed_analyser(501)

df = db.database
speed_data = list()
for i,row in tqdm(df.iterrows()):
    if isinstance(row.path2_trace_mm,str):
        trace_df = pd.read_csv(row.path2_trace_mm)
    else:
        trace_df = pd.read_csv(row.path2_head_mm)
    sa.fps = row.fps
    sa.trace_df = trace_df
    speed_data.append(sa.analyse_fish_speed_df())
speed_df = pd.concat([df[['genotype', 'sex', 'animalNo', 'expType', 'birthDate']],pd.DataFrame(speed_data)],axis=1)


import seaborn as sns
for expType in [('Ta','motivated swimming'),('Unt','free swimming')]:
    for parameter in ['thrust_mean_m/s', 'slip_mean_m/s', 'yaw_mean_m/s', 'thrust_median_m/s', 'slip_median_m/s', 'yaw_median_m/s', 
                      'cruising_thrust_mean_m/s', 'cruising_slip_mean_m/s', 'cruising_yaw_mean_m/s', 'cruising_thrust_median_m/s', 
                      'cruising_slip_median_m/s', 'cruising_yaw_median_m/s', 'activity_duration_s', 'activity_fraction', 
                      'sec_to_first_stop', 'torque']:
        f= plt.figure()
        sns.boxplot(x="genotype", y=parameter, order=['rei-INT', 'rei-HT', 'rei-HM'],# hue_order =['male','female'],
                hue="sex",data=speed_df.loc[speed_df['expType']==expType[1],:]).set_title(expType[1])
        plt.savefig('/home/bgeurten/fishDataBase/figures/'+f'{expType[1]}--{parameter}.svg'.replace(' ','_').replace('/','_per_'))
plt.show()

'''

fileDict = mff.__main__() 
#print(fileDict.keys())
dataDict = fileDict['INTF2']
expString = 'CCurr'
genName = 'rei'

reload(fishRecAnalysis)
fRAobj= fishRecAnalysis.fishRecAnalysis(dataDict,genName,expString)
fRAobj.correctionAnalysis()
#fRAobj.makeSaveFolder()
#fRAobj.saveResults()
dbEntry = fRAobj.saveDataFrames()



hist_dict = dict((k,[]) for k in db.dataBase['genotype'].unique())
for i,row in db.dataBase.iterrows():
    genotype = row['genotype']
    hist = np.genfromtxt(row['path2_probDensity'],delimiter=',')
    hist_dict[genotype].append(hist)

for key in hist_dict:
    hist_dict[key] = np.dstack(hist_dict[key] )
    norm_factor = np.nansum(np.nansum(hist_dict[key],axis=0),axis=1)
    hist_dict[key]= hist_dict[key]/norm_factor[:,None]


result = list()
for key in hist_dict:
    result.append(np.nan_to_num(np.nanmean(hist_dict[key],axis=2)))
result = dict(zip(hist_dict.keys(),result))



y_coordinates = np.linspace(0,45,9)
x_coordinates = np.linspace(0,167,18)
x,y = np.meshgrid(x_coordinates,y_coordinates)
levels = np.linspace(0.0, 5, 20)


fig, axs = plt.subplots(4, 1)
axs = axs.reshape(-1)
c = 0

for key in result:    
    im = axs[c].contourf(x, y, result[key]*1000, levels=levels)
    axs[c].set_title(key)
    axs[c].set_aspect('equal', adjustable='box')
    c+=1
plt.colorbar(im, ax=axs)
import seaborn as sns
db = fishDataBase.fishDataBase()
for parameter in ['inZoneFraction', 'inZoneDuration','inZoneMedDiverg_Deg']:
    plt.figure()
    sns.boxplot(x="genotype", y=parameter,
            hue="sex", data=db.dataBase)

plt.show()




df = db.dataBase
df_unt = df.loc[df.exp]
'''