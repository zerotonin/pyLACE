import scipy.io
import numpy as np
class matLabResultLoader():
    """
    A class to load MATLAB result files and extract relevant data.

    Attributes
    ----------
    filePosition : str
        The file path of the MATLAB result file.
    mode : str, optional
        The mode for loading the MATLAB result file, default is 'anaMat'.

    Methods
    -------
    readAnaMatFile():
        Reads the MATLAB analysis file.
    ndArray2npArray2D(ndArray):
        Converts a MATLAB array to a NumPy 2D array.
    flattenNDarray(ndArray):
        Flattens a MATLAB array and converts it to a list of NumPy arrays.
    splitResults2Variables():
        Splits the loaded results into separate variables.
    getData():
        Returns the extracted data from the MATLAB result file.
    """


    def __init__(self, filePosition, mode ='anaMat'):
        self.filePosition = filePosition
        self.mode         = mode

    def readAnaMatFile(self): 
        """
        Reads the MATLAB analysis file and extracts the data.
        """
        mat = scipy.io.loadmat(self.filePosition)
        self.metaData     = mat['metaData']
        self.analysedData = mat['analysedData']
        self.traceResult  = self.analysedData[0][0][0]

    def ndArray2npArray2D(self,ndArray):
        """
        Converts a MATLAB array to a NumPy 2D array.

        Parameters
        ----------
        ndArray : array-like
            The MATLAB array to be converted.

        Returns
        -------
        np_array : numpy.ndarray
            The converted NumPy 2D array.
        """
        temp = ndArray.tolist()
        return np.fliplr(np.array([x[0][:] for x in temp])) # fliplr as x should be first

    def flattenNDarray(self,ndArray):
        """
        Flattens a MATLAB array and converts it to a list of NumPy arrays.

        Parameters
        ----------
        ndArray : array-like
            The MATLAB array to be flattened.

        Returns
        -------
        list_of_arrays : list
            The list of flattened NumPy arrays.
        """
        temp = ndArray.tolist()
        return [np.fliplr(np.array(x[0][0].tolist())) for x in temp] # fliplr as x should be first
    
    def splitResults2Variables(self):
        """
        Splits the loaded results into separate variables.
        
        # traceInfo
        # 
        # col  1: x-position in pixel
        # col  2: y-position in pixel
        # col  3: major axis length of the fitted ellipse
        # col  4: minor axis length of the fitted ellipse
        # col  5: ellipse angle in degree
        # col  6: quality of the fit
        # col  7: number of animals believed in their after final evaluation
        # col  8: number of animals in the ellipse according to surface area
        # col  9: number of animals in the ellipse according to contour length
        # col 10: is the animal close to an animal previously traced (1 == yes)
        # col 11: evaluation weighted mean
        # col 12: detection quality [aU] if
        # col 13: correction index, 1 if the area had to be corrected automatically
        """
        self.traceInfo        = self.ndArray2npArray2D(self.traceResult[:,0])
        self.traceContour     = self.flattenNDarray(self.traceResult[:,1])
        self.traceMidline     = self.flattenNDarray(self.traceResult[:,2])
        self.traceHead        = self.ndArray2npArray2D(self.traceResult[:,3])
        self.traceTail        = self.ndArray2npArray2D(self.traceResult[:,4])
        self.trace            = self.analysedData[0][0][1]
        self.bendability      = [x[0][:] for x in self.analysedData[0][0][2].tolist()]
        self.binnedBend       = self.analysedData[0][0][3]
        self.saccs            = self.analysedData[0][0][4]
        self.trigAveSacc      = self.analysedData[0][0][5]
        self.medMaxVelocities = self.analysedData[0][0][6] 

    def getData(self):
        """
        Returns the extracted data from the MATLAB result file.

        Returns
        -------
        data : tuple
            A tuple containing the extracted data variables.
        """
        if self.mode == 'anaMat':
            self.readAnaMatFile()
            self.splitResults2Variables()
            return self.traceInfo, self.traceContour, self.traceMidline, self.traceHead, self.traceTail, self.trace, self.bendability, self.binnedBend, self.saccs, self.trigAveSacc, self.medMaxVelocities
        else:
            raise ValueError('Unknown mode for matLabResultLoader: '+str(self.mode))
