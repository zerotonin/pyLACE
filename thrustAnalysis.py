import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from tqdm import tqdm
import os,matLabResultLoader,glob

# load metaData
collectionDir = '/media/gwdg-backup/BackUp/Zebrafish/combinedData/traceResultsAna/ABTLF'
metaData = pd.read_pickle("./dummy.pkl")

#get data set
ABTLFmeta = metaData.loc[metaData['genoType'] == 'ABTLF']
ABTLFmeta = ABTLFmeta.loc[ABTLFmeta['experimentType'] == 'unmotivated']

#extract thrust
maleThrust = list()
femaleThrust = list()
for i in tqdm(range(ABTLFmeta.shape[0])):
    fileNames = glob.glob(os.path.join(collectionDir,ABTLFmeta['matFileName'].iloc[i][:-4])+'*')
    mrl = matLabResultLoader.matLabResultLoader(fileNames[0])
    traceInfo, traceContour, traceMidline, traceHead, traceTail, trace, bendability, binnedBend, saccs, trigAveSacc, medMaxVelocities = mrl.getData()
    if ABTLFmeta['sex'].iloc[i] == 'female':
        femaleThrust.append(trace[:,3])
    else:
        maleThrust.append(trace[:,3])

# get thrust triggered average

thrustThresh = 0.2
fps =200
before = 20
after  = 100
interIndMean = list()
interIndSD   = list()

for listData  in [maleThrust,femaleThrust]:
        
    meanThrusts = list()
    for thrust in listData:

        peaks,_ = find_peaks(thrust,height=thrustThresh,distance=100)
        trigAve = list()
        for peak in peaks:
            if peak > before and peak < thrust.shape[0]-after:
                trigAve.append(thrust[peak-before:peak+after])
        if trigAve:
            trigAve = np.array(trigAve)
            meanThrusts.append(np.nanmean(trigAve,axis=0))
    meanThrusts = np.array(meanThrusts)
    interIndMean.append(np.nanmean(meanThrusts,axis=0))
    interIndSD.append(np.nanstd(meanThrusts,axis=0))

# plot
x = np.linspace(-1*before/fps*1000,after/fps*1000,before+after)
plt.plot(x, interIndMean[0])
plt.fill_between(x, interIndMean[0]-interIndSD[0], interIndMean[0]+interIndSD[0],alpha=.5)
plt.plot(x, interIndMean[1])
plt.fill_between(x, interIndMean[1]-interIndSD[1], interIndMean[1]+interIndSD[0],alpha=.5)
plt.legend(['male','female'])
plt.xlabel('time, ms')
plt.ylabel('thrust, m*s-1')
plt.title('thrust triggered average')
plt.show()


