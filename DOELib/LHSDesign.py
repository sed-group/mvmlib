from pyDOE import lhs

def scaling(x,l,u,operation):
    """
    Scaling by a range

    Inputs
    ------
    x : np.array
        2d array of size n * nsamples of datapoints
    l : np.array
        1d array of length n specifying lower range of features
    u : np.array
        1d array of length n = len(l) specifying upper range of features
    operation : int
        The flag type indicates whether to scale (1) or unscale (2)

    Outputs
    -------
    x_out : np.array
        2d array of size n * nsamples of unscaled datapoints
    
    """
    
    if operation == 1:
        # scale
        x_out=(x-l)/(u-l)
    elif operation == 2:
        # unscale
        x_out = l + x*(u-l)
    
    return x_out

class LHSDesign():

    def __init__(self,lb,ub,nsamples,criterion='maximin'):
        """
        Constructor

        Inputs
        ------
        lb : int
            1d array of length n specifying lower range of features
        ub : str
            1d array of length n = len(lb) specifying upper range of features
        nsamples : int
            The number of samples to generate for each factor (Default: n)

        Optional
        --------
        criterion : str
            Allowable values are "center" or "c", "maximin" or "m", 
            "centermaximin" or "cm", and "correlation" or "corr". If no value 
            given, the design is simply randomized.

        TODO: return error if len(lb) != len(ub)

        """

        self.lb = lb
        self.ub = ub

        self.nsamples = nsamples
        self.criterion = criterion

        # Generate latin hypercube design and store it
        self.design_LH = lhs(len(self.lb), samples=self.nsamples, criterion=self.criterion)

    def unscale(self): 
        """
        Unscale latin hypercube by ub and lb

        Outputs
        -------
        unscaled_LH: np.array
            numpy array of size n * nsamples of LH values unscaled by lb and ub

        """

        unscaled_LH = scaling(self.design_LH,self.lb,self.ub,2)
        
        return unscaled_LH

    def scale(self): 
        """
        return scaled latin hypercube between 0 and 1

        Outputs
        -------
        scaled_LH: np.array
            numpy array of size n * nsamples of LH values between 0 and 1

        """
        
        return self.design_LH