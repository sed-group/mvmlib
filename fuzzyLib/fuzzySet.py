    
import skfuzzy as fuzz

class fuzzySet():

    def __init__(self,lo,md,hi,label=''):

        """
        Constructor

        Inputs
        ------
        lo,md,hi: fuzzyFunction
            instances of fuzzyFunction class

        Optional
        --------
        label: str
            string to tag instance with
        """

        self.lo = lo
        self.md = md
        self.hi = hi

        self.universe = lo.universe
        self.label = label

        # TODO: if different universes raise an error

    def interp(self,input):

        """
        Interpret membership of input

        Inputs
        ------
        input: float
        """

        level_lo = self.lo.interp(input) 
        level_md = self.md.interp(input)
        level_hi = self.hi.interp(input)

        return level_lo,level_md,level_hi
