    
import skfuzzy as fuzz

class fuzzyFunction():

    def __init__(self,universe,label=''):
        """
        Contrains description and implementation of
        different fuzzy membership functions
        
        Parameters
        ----------
        universe : np.array
            1d array of length n
            n=number of samples
        label : str, optional
            string to tag instance with
        """

        self.universe = universe
        self.label = label

class triangularFunc(fuzzyFunction):

    def __init__(self, universe,label=''):
        super().__init__(universe,label)

    def setFunc(self,low,medium,high):
        """
        Create an instance of the triangular membership function

        Parameters
        ----------
        low : int
            left foot of triangle
        medium : int
            peak of triangle
        high : int
            right foot of triangle
        """

        self.low = low; self.medium = medium; self.high = high

    def setLabel(self,label):
        """
        Changes label property of function

        Parameters
        ----------
        label : float
            new label name
        """

        self.label=label

    def getArray(self):
        """
        Sample an array of values from membership function
        
        Returns
        -------
        array : np.array
            1d array of length universe
        """

        self.array = fuzz.trimf(self.universe, [self.low, self.medium, self.high])
        return self.array

    def interp(self,input=None):
        """
        Interpret membership of input ot fuzzy function

        Parameters
        ----------
        input : float

        Returns
        -------
        level : float
            interpreted membership level of the input
        """

        if input:
            level = fuzz.interp_membership(self.universe, self.getArray(), input)
        else:
            level = fuzz.interp_membership(self.universe, self.getArray(), self.universe[0])

        return level