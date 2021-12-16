import numpy as np
from typing import Dict, Any, AnyStr, List, Type, Union
import matplotlib.pyplot as plt
from .uncertaintyLib import Distribution,gaussianFunc
from .utilities import check_folder

"""Design margins library for computing buffer and excess"""

class FixedParam():
    def __init__(self,value:Union[float,int,str],key:str,
        description:str='',symbol:str=''):
        """
        Contains description of an input parameter to the MAN
        is inherited by DesignParam

        Parameters
        ----------
        value : Union[float,int,str]
            the value of the input spec
        key : str
            unique identifier
        description : str, optional
            description string, by default ''
        symbol : str, optional
            shorthand symbol, by default ''
        """

        self.value          = value
        self.description    = description
        self.symbol         = symbol
        self.key            = key
        self.type           = type(value)

    def __call__(self) -> Union[float,int,str]:
        """
        retrieve the value of the parameter

        Returns
        -------
        Union[float,int,str]
            The value of the parameter
        """

        return self.value # return the requested value

class DesignParam(FixedParam):
    def __init__(self,value:Union[float,int,str],key:str,
        universe:Union[tuple,list],description:str='',symbol:str=''):
        """
        Contains description of an input parameter to the MAN
        is inherited by DesignParam, and FixedParam

        Parameters
        ----------
        value : Union[float,int,str]
            the value of the input spec
        key : str
            unique identifier
        universe : Union[tuple,list]
            the possible values the design parameter can take, 
            If tuple must be of length 2 (upper and lower bound)
            type(value) must be float, or int
        description : str, optional
            description string, by default ''
        symbol : str, optional
            shorthand symbol, by default ''
        """
        super().__init__(value,key,description,symbol)
        
        if type(universe) == tuple:
            assert len(universe) == 2
            assert self.type in [float,int]
        elif type(universe) == list:
            assert len(universe) > 0

        self.universe = universe

class InputSpec():
    def __init__(self,value: Union[float,int,Distribution,gaussianFunc],
        key: str,description:str='',symbol:str='',cov_index=0):
        """
        Contains description of an input specification
        could deterministic or stochastic
        
        Parameters
        ----------
        value : Union[float,int,Distribution,gaussianFunc]
            the value of the input spec, 
            if type is Distribution then a sample is drawn
        key : str
            unique identifier
        description : str, optional
            description string, by default ''
        symbol : str, optional
            shorthand symbol, by default ''
        cov_index : int, optional
            which random variable to draw from 
            if multivariate distribution is provided, by default 0
        """

        self.value          = value
        self.description    = description
        self.symbol         = symbol
        self.key            = key
        self.cov_index      = cov_index
        self.type           = type(value)
        self._samples       = np.empty(0)

        # Check if input spec if stochastic
        if type(value) in [Distribution,gaussianFunc]:
            self.stochastic = True
        elif type(value) in [float,int]:
            self.stochastic = False

        # Check if input spec is co-dependant on another
        if type(value) == gaussianFunc:
            self.ndim = value.ndim
        else:
            self.ndim = 1

        assert self.cov_index <= self.ndim

    @property
    def samples(self) -> np.ndarray:
        """
        sample vector getter

        Returns
        -------
        np.ndarray
            vector of sample observations
        """
        return self._samples # return 

    @samples.setter
    def samples(self,s:Union[float,np.ndarray]):
        """
        Appends value observation s to sample vector

        Parameters
        ----------
        s : Union[float,np.ndarray]
            value to append to sample vector
        """
        self._samples = np.append(self._samples,s)

    def reset(self):
        """
        Resets the stored samples
        """
        self._samples = np.empty(0)

    def view(self,xlabel='',savefile=None):
        """
        view 1D or 2D plot of probability distribution of excess
        """

        self.fig, self.ax = plt.subplots(figsize=(8, 3))
        self.ax.hist(self.samples, bins=100, density=True)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel('density')

        if savefile is not None:
            # Save figure to image
            check_folder('images/%s' %(self.key))
            self.fig.savefig('images/%s/%s.pdf' %(self.key,savefile), 
                format='pdf', dpi=200, bbox_inches='tight')

        plt.show()

    def __call__(self,N=1):
        """
        draw random samples from value

        Parameters
        ----------
        N : int, optional
            Number of random samples to draw
            default is one sample

        Returns
        -------
        np.ndarray
            A 1D array of size N, where 
            N is the number of requested samples
        """

        if self.stochastic:
            assert self.value.samples.shape[1] >= N
            samples = self.value.samples[self.cov_index,-N:] # retrieve last N samples
        else:
            samples = self.value * np.ones(N)
        
        self.samples = samples # store the samples inside instance
        return samples # return the requested number of samples

class Behaviour():
    def __init__(self,key=''):
        """
        This class stores the method for calculating its outputs
            - Intermediate parameters
            - Performance parameters
            - Decided values
            - target thresholds

        Parameters
        ----------
        key : str, optional
            string to tag instance with, default = ''
        """
        self.key            = key
        self.intermediate   = None
        self.performance    = None
        self.decided_value  = None
        self.threshold      = None

    def reset(self):
        """
        Resets the stored variables
        """
        self.intermediate   = None
        self.performance    = None
        self.decided_value  = None
        self.threshold      = None

    def __call__(self, *args, **kwargs):
        """
        The function that will be used to calculate the outputs of the behaviour model
            - Can be a deterministic model
            - Can be a stochastic model (by calling a defined dmLib.Distribution instance)
        This method must be redefined by the user for every instance

        Example
        -------
        >>> # [in the plugin file]
        >>> from dmLib import Behaviour
        >>> class myBehaviour(Behaviour):
        >>>     def behaviour(self,r,d):
        >>>         # some specific model-dependent behaviour
        >>>         self.intermediate   = d
        >>>         self.performance    = r*2+1 / d
        >>>         self.decided_value  = r**2
        >>>         self.threshold      = r/d
        """
        # default code for the default behaviour
        return

class MarginNode():
    def __init__(self,key:str='',cutoff:float=0.9,buffer_limit:int=0,type:str='must_exceed'):
        """
        Contains description and implementation 
        of a Margin Node object which is the building block
        of a Margin Analysis Network (MAN)

        Parameters
        ----------
        cutoff : float, optional
            cutoff limit for calculating reliability,
            default = 0.9
        buffer_limit : float, optional
            lower bound for beginning of buffer zone,
            default = 0.0
        key : str, optional
            string to tag instance with, default = ''
        type : str, optional
            possible values('must_exceed','must_not_exceed'), by default 'must_exceed'
        """

        self.key            = key
        self.cutoff         = cutoff
        self.buffer_limit   = buffer_limit
        self.type           = type
        self._target        = np.empty(0)
        self._decided_value = np.empty(0)
        self._excess        = np.empty(0)
        self._excess_dist   = None

        assert self.type in ['must_exceed','must_not_exceed']

    @property
    def target(self):
        """
        Target vector getter

        Returns
        -------
        np.1darray
            vector of target observations
        """
        return self._target

    @target.setter
    def target(self,t):
        """
        Appends target observation t to target vector

        Parameters
        ----------
        t : float OR np.1darray
            value to append to target vector
        """
        self._target = np.append(self._target,t)

    @property
    def decided_value(self):
        """
        Response vector getter

        Returns
        -------
        np.1darray
            vector of response observations
        """
        return self._decided_value

    @decided_value.setter
    def decided_value(self,r):
        """
        Appends response observation r to target vector

        Parameters
        ----------
        r : float OR np.1darray
            value to append to response vector
        """
        self._decided_value = np.append(self._decided_value,r)

    @property
    def excess(self):
        """
        Excess vector getter

        Returns
        -------
        np.1darray
            vector of excess observations
        """
        return self._excess

    @excess.setter
    def excess(self,e):
        """
        Appends excess observation e to target vector

        Parameters
        ----------
        e : float OR np.1darray
            value to append to response vector
        """
        self._excess = np.append(self._excess,e)

    @property
    def excess_dist(self) -> Distribution:
        """
        Excess Distribution object

        Returns
        -------
        dmLib.Distribution
            instance of dmLib.Distribution holding excess pdf
        """
        return self._excess_dist

    @excess_dist.setter
    def excess_dist(self,excess):
        """
        Creates excess Distribution object

        Parameters
        ----------
        excess : np.1darray
            Vector of excess values
        """
        self._excess_dist = Distribution(excess, lb=min(excess),ub=max(excess))

    def reset(self):
        """
        Resets accumulated random observations in target, 
        response, and excess attributes
        """
        self._target = np.empty(0)
        self._decided_value = np.empty(0)
        self._excess = np.empty(0)
        self._excess_dist = None

    def compute_cdf(self,bins=500):
        """
        Calculate the cumulative distribution function for the excess margin

        Parameters
        ----------
        bins : int, optional
            number of discrete bins used to 
            construct pdf and pdf curves, by default 500

        Returns
        -------
        bin_centers : np.1darray
            array of len(excess) - 1 containing the x-axis values of CDF
        cdf : np.1darray
            array of len(excess) - 1 containing the y-axis values of CDF
        excess_limit : float
            The value on the x-axis that corresponds to the cutoff probability
        reliability : float
            the probability of excess being >= target
        """
        def moving_average(x, w):
            """
            N-moving average over 1D array

            Parameters
            ----------
            x : np.1darray
                input array to average
            w : int
                number of elements to average

            Returns
            -------
            np.1darray
                avaeraged array
            """
            return np.convolve(x, np.ones(w), 'valid') / w

        excess_hist = np.histogram(self.excess, bins=bins,density=True)
        bin_width = np.mean(np.diff(excess_hist[1]))
        bin_centers = moving_average(excess_hist[1],2)
        cdf = np.cumsum(excess_hist[0] * bin_width)

        excess_limit = bin_centers[cdf >= self.cutoff][0]
        reliability = 1 - cdf[bin_centers >= self.buffer_limit][0]

        return bin_centers, cdf, excess_limit, reliability

    def view(self,xlabel='',savefile=None):
        """
        view 1D or 2D plot of probability distribution of excess
        """

        self.fig, self.ax = plt.subplots(figsize=(8, 3))
        self.ax.hist(self.excess, bins=100, density=True)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel('density')

        if savefile is not None:
            # Save figure to image
            check_folder('images/%s' %(self.key))
            self.fig.savefig('images/%s/%s.pdf' %(self.key,savefile), 
                format='pdf', dpi=200, bbox_inches='tight')

        plt.show()

    def view_cdf(self,xlabel='',savefile=None):
        """
        view 1D or 2D plot of cumulative distribution of excess
        """

        # calculate CDF
        bin_centers, cdf, excess_limit, reliability = self.compute_cdf()
        buffer_band = (bin_centers >= self.buffer_limit) & (cdf <= self.cutoff)
        excess_band = cdf >= self.cutoff

        self.figC, self.axC = plt.subplots(figsize=(8, 3))
        self.axC.plot(bin_centers, cdf, '-b')
        self.axC.vlines([self.buffer_limit,excess_limit],[0,0],[1-reliability,self.cutoff],linestyles='dashed')
        self.axC.fill_between(bin_centers[buffer_band], 0, cdf[buffer_band], facecolor='Green', alpha=0.4, label='Buffer')
        self.axC.fill_between(bin_centers[excess_band], 0, cdf[excess_band], facecolor='Red', alpha=0.4, label='Excess')
        
        tb = self.axC.text((excess_limit + self.buffer_limit) / 2 - 0.2, 0.1, 'Buffer', fontsize=14)
        te = self.axC.text((excess_limit + bin_centers[-1]) / 2 - 0.2, 0.1, 'Excess', fontsize=14)
        tb.set_bbox(dict(facecolor='white'))
        te.set_bbox(dict(facecolor='white'))

        self.axC.set_xlabel(xlabel)
        self.axC.set_ylabel('Cumulative density')

        if savefile is not None:
            # Save figure to image
            check_folder('images/%s' %(self.key))
            self.figC.savefig('images/%s/%s.pdf' %(self.key,savefile), 
                format='pdf', dpi=200, bbox_inches='tight')

        plt.show()

    def __call__(self,decided_value:np.ndarray,target_threshold:np.ndarray):
        """
        Calculate excess given the target threshold and decided value

        Parameters
        ----------
        decided_value : np.1darray
            decided values to the margin node describing the capability of the design.
            The length of this vector equals the number of samples
        target_threshold : np.1darray
            The target threshold parameters that the design needs to achieve
            The length of this vector equals the number of samples
        """
            
        self.decided_value = decided_value # add to list of decided values
        self.target = target_threshold # add to list of targets

        if self.type == 'must_exceed':
            e = decided_value - target_threshold
        elif self.type == 'must_not_exceed':
            e = target_threshold - decided_value
        else:
            raise Exception('Wrong margin type (%s) specified. Possible values are "must_Exceed", "must_not_exceed".' %(str(self.type)))
        self.excess = e # add to list of excesses

class MarginNetwork():
    def __init__(self,design_params:List[DesignParam],input_specs:List[InputSpec],
        fixed_params:List[FixedParam],behaviours:List[Behaviour],
        margin_nodes:List[MarginNode],key:str=''):
        """
        The function that will be used to calculate a forward pass of the MAN
        and associated metrics of the MVM
            - The first metric is change absorption capability (CAC)
            - The second metric is the impact on performance (IoP)
        This class should be inherited and the forward method should be implemented by the user
        to describe how the input params are related to the output params (DDs,TTs, and performances)

        Parameters
        ----------
        design_params : List[DesignParam]
            list of DesignParam instances 
        input_specs : List[InputSpec]
            list of InputSpec instances 
        fixed_params : List[FixedParam]
            list of FixedParam instances 
        behaviours : List[Behaviour]
            list of Behaviour instances
        margin_nodes : List[MarginNode]
            list of MarginNode instances
        key : str, optional
            string to tag instance with, default = ''
        """
        self.design_params  = design_params
        self.input_specs    = input_specs
        self.fixed_params   = fixed_params
        self.behaviours     = behaviours
        self.margin_nodes   = margin_nodes
        self.key            = key

    def forward(self, *args, **kwargs):
        """
        The function that will be used to calculate a forward pass of the MAN
        (design_params,input_specs,fixed_params) -> (excess,performance)
        This method must be redefined by the user for every instance

        Example
        -------
        >>> # [in the plugin file]
        >>> from dmLib import MarginNetwork
        >>> class myMarginNetwork(MarginNetwork):
        >>>     def forward(self):
        >>>         # some specific model-dependent behaviour
        >>>         d1 = self.design_params[0]
        >>>         s1 = self.input_specs[0]
        >>>         p1 = self.fixed_params[0]
        >>>         b1 = self.behaviours[0]
        >>>         b2 = self.behaviours[1] # dependant on b1
        >>>         e1 = self.margin_nodes[0]
        >>>         # Execution behaviour models
        >>>         b1(d1.value,p1.value)
        >>>         b2(d1.value,b1.intermediate)
        >>>         e1(b3.decided_value,s3())
        """
        # default code for the default threshold
        return