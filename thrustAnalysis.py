import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from tqdm import tqdm
import os,matLabResultLoader,glob

def exact_mc_perm_test(xs, ys, nmc):
    n, k = len(xs), 0
    diff = np.abs(np.median(xs) - np.median(ys))
    zs = np.concatenate([xs, ys])
    for j in range(nmc):
        np.random.shuffle(zs)
        k += diff < np.abs(np.mean(zs[:n]) - np.mean(zs[n:]))
    return k / nmc


# load metaData
collectionDir = '/media/gwdg-backup/BackUp/Zebrafish/combinedData/traceResultsAna/ABTLF'
metaData = pd.read_pickle("/media/gwdg-backup/BackUp/Zebrafish/combinedData/traceResultsAna_meta_pandasPickle.pkl")

#get data set
ABTLFmeta = metaData.loc[metaData['genoType'] == 'ABTLF']
ABTLFmeta = ABTLFmeta.loc[ABTLFmeta['experimentType'] == 'tapped']

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

thrustThresh = 0.1
fps =200
before = 20
after  = 100
interIndMean = list()
thrustFreq   = list()
interIndSD   = list()

for listData  in [maleThrust,femaleThrust]:
        
    meanThrusts = list()
    freq    = list()
    for thrust in listData:

        peaks,_ = find_peaks(thrust,height=thrustThresh,distance=100)
        trigAve = list()
        for peak in peaks:
            if peak > before and peak < thrust.shape[0]-after:
                trigAve.append(thrust[peak-before:peak+after])
        if trigAve:
            freq.append(len(trigAve)/(thrust.shape[0]/fps))
            trigAve = np.array(trigAve)
            meanThrusts.append(np.nanmean(trigAve,axis=0))
    meanThrusts = np.array(meanThrusts)
    interIndMean.append(np.nanmean(meanThrusts,axis=0))
    interIndSD.append(np.nanstd(meanThrusts,axis=0))
    thrustFreq.append(np.array(freq)*60)

p = exact_mc_perm_test(interIndMean[0],interIndMean[1],20000)
p2 = exact_mc_perm_test(thrustFreq[0],thrustFreq[1],20000)
# plot
fig,ax =plt.subplots()
x = np.linspace(-1*before/fps*1000,after/fps*1000,before+after)
ax.plot(x, interIndMean[0])
ax.fill_between(x, interIndMean[0]-interIndSD[0], interIndMean[0]+interIndSD[0],alpha=.5)
ax.plot(x, interIndMean[1])
ax.fill_between(x, interIndMean[1]-interIndSD[1], interIndMean[1]+interIndSD[0],alpha=.5)
ax.legend(['male','female'])
plt.xlabel('time, ms')
plt.ylabel('thrust, m*s-1')
plt.title(f'thrust triggered average | perm. median p = {str(p)}')

fig2,ax2 =plt.subplots()
ax2.boxplot(thrustFreq)
plt.ylabel('frequency of thrust strokes, min-1')
plt.title(f'stroke frequency per minute | perm. median p = {str(p2)}')
plt.show()

