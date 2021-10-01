    
import skfuzzy as fuzz

class fuzzyFunction():
    
    """
    Contrains description and implementation of
    different fuzzy membership functions
    """

    def __init__(self,universe,label=''):
        """
        Constructor

        universe: 1d array of length n
        n=number of samples
        """

        self.universe = universe
        self.label = label

class triangularFunc(fuzzyFunction):

    def __init__(self, universe,label=''):
        super().__init__(universe,label)

    def setFunc(self,low,medium,high):
        """
        Create an instance of the triangular membership function

        low: int
        medium: int
        high: int
        """

        self.low = low; self.medium = medium; self.high = high

    def getArray(self):
        """
        Sample an array of values from membership function
        
        Output:
        1d array of length n
        """

        self.array = fuzz.trimf(self.universe, [self.low, self.medium, self.high])
        return self.array

    def interp(self,input=None):
        """
        Interpret membership of input ot fuzzy function
        input: 'float'
        """

        if input:
            level = fuzz.interp_membership(self.universe, self.getArray(), input)
        else:
            level = fuzz.interp_membership(self.universe, self.getArray(), self.universe[0])

        return level