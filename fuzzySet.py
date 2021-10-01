    
import skfuzzy as fuzz

class fuzzySet():

    def __init__(self,lo,md,hi,label=''):

        """
        Constructor

        lo,md,hi: instances of fuzzyFunction class
        !must have same universe
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

        input: float
        """

        level_lo = self.lo.interp(input) 
        level_md = self.md.interp(input)
        level_hi = self.hi.interp(input)

        return level_lo,level_md,level_hi
