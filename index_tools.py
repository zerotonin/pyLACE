import numpy as np
import copy

def bool2indice(boolList):
    """This function builds a succession of indices from a
       booolean signal
       Example:    boolean array 1,1,1,0,1,1,0,1
                          output 0,1,2,  4,5,  7


    Args:
        boolList (list): of either true or false values

    Returns:
        numpy.array: integers with the indices of true statements
    """
    return np.array([i for i, x in enumerate(boolList) if x])

def indice_seq2start_end(indiceList):
    """_summary_

    Args:
        indiceList (_type_): _description_

    Returns:
        _type_: _description_
    """
    indDiff = np.diff(indiceList)
    starts = [indiceList[0]]
    ends = list()
    for i in range(1,len(indiceList)):
        if indDiff[i-1] !=1:
            ends.append(indiceList[i-1])
            starts.append(indiceList[i])
    ends.append(indiceList[-1])
    return np.array(list(zip(starts,ends)))

def bool_Seq2start_end_indices(boolList):
    indices = bool2indice(boolList)
    return indice_seq2start_end(indices)

def bracket_bools(boolList):
    returnList = copy.deepcopy(boolList)
    for i in range(1,len(returnList)-1):
        if boolList[i] == 1:
            returnList[i-1],returnList[i+1] = (True,True)
    return returnList

def bracket_starts_end_of_sequence(startEndSequenceInd,seqLen):
    returnList = copy.deepcopy(startEndSequenceInd)
    for seqI in range(returnList.shape[0]):
        
        if returnList[seqI,0] != 0 and returnList[seqI,1] >= seqLen:
            returnList[seqI,0] = returnList[seqI,0]-1
            returnList[seqI,1] = returnList[seqI,1]+1
    return returnList

def get_duration_from_start_end(startEndInd):
    return np.diff(startEndInd)
    
