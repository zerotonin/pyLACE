from importlib import reload
import numpy as np
import pandas as pd
from data_handlers.mediaHandler import mediaHandler
from traceCorrector import traceCorrector
from fish_data_base.counterCurrentAna import sortMultiFileFolder
import traceAnalyser
import fishPlot
import matplotlib.pyplot as plt
import fish_data_base.fishRecAnalysis as fishRecAnalysis
import fish_data_base.fishDataBase as fishDataBase
import seaborn as sns

multiFileFolder = '/media/gwdg-backup/BackUp/Vranda/data_counter_c-start/countercurrent_onefolder/rei_last_generation_11-2018'
multiFileFolder = '/media/gwdg-backup/BackUp/Vranda/Finaldata_rei/Motivated_trials_rei'
db = fishDataBase.fishDataBase()
# Experiment types CCur counter current , Ta tapped, Unt untapped, cst, c-startz
db.run_multi_trace_folder(multiFileFolder,'rei','Ta','11-2018',start_at=0)



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



hist_dict = dict((k,[]) for k in db.database['genotype'].unique())
for i,row in db.database.iterrows():
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
    sns.boxplot(x="genotype", y=parameter, order=['rei-INT', 'rei-HT', 'rei-HM'],
            hue="sex", data=db.database)

plt.show()



df = db.database
df_unt = df.loc[df.exp]