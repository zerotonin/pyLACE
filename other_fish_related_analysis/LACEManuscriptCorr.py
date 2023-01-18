from enum import auto
from os import minor
import data_handlers.matLabResultLoader as matLabResultLoader
from importlib import reload
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import fishPlot,os
import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import glob

sourceDir = '/media/gwdg-backup/BackUp/Zebrafish/combinedData/traceResultsAna/'
savePos   = '/media/dataSSD/zebraFischTrackingExample/fishDf.h5'
files = [os.path.join(sourceDir,x) for x in glob.iglob(sourceDir + '**/*.mat')]

results = list()

for file in tqdm(files,desc='read matlab files'):
    mrl = matLabResultLoader.matLabResultLoader(file)
    mrl.getData()
    autoCorrector = []
    quality = []

    for entry in mrl.traceResult:
        quality.append(entry[0][0][11])
        autoCorrector.append(entry[0][0][12])

    mQual    = np.mean(quality)
    numCorr  = np.sum(autoCorrector)
    frames   = len(autoCorrector)
    freqCorr = numCorr/frames
    results.append([frames, numCorr,freqCorr,mQual])

df = pd.DataFrame(np.array(results),columns=['frame','number of corrections','corrections per frame','median quality, au'])
df.to_hdf(savePos,key='df')
#sns.distplot(df["corrections per frame"],kde=False)
sns.displot(df, x="corrections per frame", binwidth=.01,kde=False,rug=True, log_scale=(False, True))
#ax = sns.distplot(df["corrections per frame"], rug=True, rug_kws={"color": "g"},
#                  kde_kws={"color": "k", "lw": 3, "label": "KDE"},
#                  hist_kws={"histtype": "step", "linewidth": 3,
#                            "alpha": 1, "color": "g"})

plt.show()