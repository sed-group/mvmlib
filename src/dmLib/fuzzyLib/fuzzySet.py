    
import skfuzzy as fuzz

class fuzzySet():

    def __init__(self,lo,md,hi,label=''):

        """
        Constructor

        Parameters
        ----------
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

        self.lo.setLabel(label)
        self.md.setLabel(label)
        self.hi.setLabel(label)

        # TODO: if different universes raise an error


    def setLabel(self,label):
        """
        Changes label property of fuzzy set and all its membership functions

        Parameters
        ----------
        label: float
            new label name

        """

        self.label=label
        self.lo.setLabel(label)
        self.md.setLabel(label)
        self.hi.setLabel(label)

    def interp(self,input):

        """
        Interpret membership of input

        Parameters
        ----------
        input: float
        """

        level_lo = self.lo.interp(input) 
        level_md = self.md.interp(input)
        level_hi = self.hi.interp(input)

        return level_lo,level_md,level_hi
