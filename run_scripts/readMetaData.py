import scipy.io, os,re
import pandas as pd

mat = scipy.io.loadmat('/media/gwdg-backup/BackUp/Zebrafish/combinedData/traceResultsAna_meta.mat')

metaAll = mat['metaAll']
matFileName = list()
genoType = list()
sex = list()
experimentType = list()
pix2mm = list()
bwFilterSet =list()
fps = list()
saccThresh = list()
dateTime = list()

for fileI in range(metaAll.shape[1]):
    file = metaAll[0,fileI]
    matFileName.append(os.path.basename(file[1][0]))
    genoType.append(file[2][0])
    sex.append(file[3][0])
    experimentType.append(file[4][0])
    pix2mm.append(file[5][0])
    bwFilterSet.append(file[6][0])
    fps.append(file[7][0])
    saccThresh.append(file[8][0])
    dateTime.append([int(s) for s in re.findall(r'\d+', os.path.basename(file[1][0]))])

metaDict = {'dateTime':dateTime,'sex':sex,'genoType':genoType,'experimentType':experimentType,'fps':fps,'pix2mm':pix2mm,'bwFilterSet':bwFilterSet,'saccThresh':saccThresh,'matFileName':matFileName}


metaData = pd.DataFrame(metaDict)
metaData.to_csv('/media/gwdg-backup/BackUp/Zebrafish/combinedData/traceResultsAna_meta.csv', index=False)
metaData.to_pickle('/media/gwdg-backup/BackUp/Zebrafish/combinedData/traceResultsAna_meta_pandasPickle.pkl')

ABTLFmeta = metaData.loc[metaData['genoType'] == 'ABTLF']
