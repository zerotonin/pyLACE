import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
from mlxtend.evaluate import permutation_test

def getCI(data):
    data = data[np.logical_not(np.isnan(data))]
    return st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))

def getCIfrom2D(data,axis=0):
    if axis == 0:
        shapeRange = 1
    else:
        shapeRange = 0
    CI = list()
    for colI in range(data.shape[shapeRange]):
        CI.append(getCI(data[:,colI]))
    return CI

def readFile(fPos):
    df   = pd.read_csv(fPos, index_col=0, header=None).T
    dataLabels = list(df.columns) 
    return df.to_numpy(),dataLabels

def scatterplot(data,med,CI,dataLabels,ylim=(0,30.)):
    for setI in range(data.shape [1]):
        plt.scatter(np.ones(shape= data.shape[0])*setI,data[:,setI])
        ci = [[CI[setI][0]],[CI[setI][1]]]
        plt.errorbar(setI, med[setI], yerr=ci, marker='s',
         mec='k')
        
    ax = plt.gca()
    ax.set_xticks(np.linspace(0,data.shape[1]-1,data.shape[1]))
    ax.set_xticklabels(dataLabels, rotation = 45, ha="right")
    plt.ylabel(r"Cholestrol, µg")
    plt.ylim(ylim)

def addSignificance(xCoords,yCoords,offSet,significanceStr = 'ns', colorStr='k'):    
    import matplotlib.pyplot as plt
    import numpy as np

    # statistical annotation
    x1, x2 = xCoords   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
    y1, y2 = yCoords
    y1, y2 = y1+offSet,y2+offSet
    y3, h  = np.max(yCoords) + 2*offSet, 2*offSet, 
    plt.plot([x1, x1, x2, x2], [y1, y3+h, y3+h, y2], lw=1.5, c=colorStr)
    plt.text((x1+x2)*.5, y3+h, significanceStr, ha='center', va='bottom', color=colorStr)

def fisherTest4Fluo(groupA, groupB):
        # How does python interpret the Fisher test: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html
    # What's Fishers exact test: https://en.wikipedia.org/wiki/Fisher%27s_exact_test
    import scipy.stats as stats
    oddsratio, pvalue = stats.fisher_exact([[groupA[0], groupB[0]], [groupA[1]-groupA[0], groupB[1]-groupB[0]]])
    return oddsratio,pvalue

data,dataLabels = readFile('run_01_AfterFreezeDryNorm.csv')
med  = np.nanmedian(data,axis=0)
CI   = getCIfrom2D(data,axis=0)

plt.subplot(1,3,1)
scatterplot(data[:,0:3],med[0:3],CI[0:3],dataLabels[0:3])

plt.subplot(1,3,2)
scatterplot(data[:,3:6],med[3:6],CI[3:6],dataLabels[3:6])
plt.subplot(1,3,3)
scatterplot(data[:,6:9],med[6:9],CI[6:9],dataLabels[6:9])
plt.show()