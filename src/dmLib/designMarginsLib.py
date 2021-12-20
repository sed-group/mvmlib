import numpy as np
from typing import Dict, Any, AnyStr, Tuple, List, Union
import matplotlib.pyplot as plt
from smt.surrogate_models import KRG

from .uncertaintyLib import Distribution,gaussianFunc,VisualizeDist
from .DOELib import Design
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

        self.description    = description
        self.symbol         = symbol
        self.key            = key
        self.type           = type(value)
        self._value         = value

    @property
    def value(self) -> float:
        """
        returns the stored value

        Returns
        -------
        float
            current value
        """
        return self._value

    @value.setter
    def value(self,v:float):
        """
        adjusts the current value

        Parameters
        ----------
        v : float
            value to set the parameter to
        """

        self._value = v

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
        self.original   = value
        self.universe   = universe

    def reset(self):
        """
        resets the design parameters to its initial value given at `__init__`   
        """
        self.value = self.original

class InputSpec(VisualizeDist):
    def __init__(self,value:Union[float,int,Distribution,gaussianFunc],
        key: str,description:str='',symbol:str='',distribution:Distribution=None, 
        cov_index:int=0, inc:float=5.0, inc_type:str='rel'):
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
        distribution : Distribution, optional
            if a Distribution object is provided, then the spec can sampled by calling it
            Example:
            >>> from dmLib import InputSpec, gaussianFunc
            >>> dist = gaussianFunc(1.0, 0.1)
            >>> s1 = InputSpec(1.0, 'S1', dist)
            >>> sample = s1()
        cov_index : int, optional
            which random variable to draw from 
            if multivariate distribution is provided, by default 0
        inc : float, optional
            The value by which to increment the input specification during deterioration calculation
            (can be negative), by default 5.0
        inc_type : str, optional
            possible values ('rel','abs')
            if 'rel' then the increment is multiplied by the nominal value, 
            if 'abs' then the increment is applied directly on the input spec, by default 'rel'
        """

        self.description    = description
        self.symbol         = symbol
        self.key            = key
        self.distribution   = distribution
        self.cov_index      = cov_index
        self.inc            = inc
        self.inc_type       = inc_type
        self.original       = value

        self._value         = value
        self._values        = np.empty(0)

        # Check if input spec if stochastic
        if type(distribution) in [Distribution,gaussianFunc]:
            self.stochastic = True
        else:
            self.stochastic = False

        # Check if input spec is co-dependant on another
        if type(value) == gaussianFunc:
            self.ndim = value.ndim
        else:
            self.ndim = 1

        assert self.cov_index <= self.ndim

    @property
    def values(self) -> np.ndarray:
        """
        sample vector getter

        Returns
        -------
        np.ndarray
            vector of sample observations
        """
        return self._values # return 

    @values.setter
    def values(self,v:Union[float,np.ndarray]):
        """
        Appends value observation v to sample vector

        Parameters
        ----------
        v : Union[float,np.ndarray]
            value to append to sample vector
        """
        self._values = np.append(self._values,v)

    @property
    def value(self) -> float:
        """
        Value getter

        Returns
        -------
        float
            Last value observation
        """

        return self._value # return the requested stored value

    @value.setter
    def value(self,v:float):
        """
        Value observation v to store

        Parameters
        ----------
        v : float
            Value observation to store
        """
        self._value = v

    def reset(self):
        """
        Resets the stored samples
        """
        self._value     = self.original

    def view(self,xlabel:str=None,folder:str='',file:str=None,img_format:str='pdf'):
        """
        Views the distribution of the performance parameter

        Parameters
        ----------
        xlabel : str, optional
            axis label of value , if not provided uses the key of the object, 
            by default None
        folder : str, optional
            folder in which to store image, by default ''
        file : str, optional
            name of image file, if not provide then an image is not saved, by default None
        img_format : str, optional
            format of the image to be stored, by default 'pdf'
        """
        if xlabel is None:
            xlabel = '%s' %self.key
        super().view(xlabel=xlabel,folder=folder,file=file,img_format=img_format)

    def view_cdf(self,xlabel:str=None,folder:str='',file:str=None,img_format:str='pdf'):
        """
        Views the cumulative distribution of the performance parameter

        Parameters
        ----------
        xlabel : str, optional
            axis label of value , if not provided uses the key of the object, 
            by default None
        folder : str, optional
            folder in which to store image, by default ''
        file : str, optional
            name of image file, if not provide then an image is not saved, by default None
        img_format : str, optional
            format of the image to be stored, by default 'pdf'
        """
        if xlabel is None:
            xlabel = '%s' %self.key
        super().view_cdf(xlabel=xlabel,folder=folder,file=file,img_format=img_format)

    def __call__(self,N:int=1) -> np.ndarray:
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
            assert self.distribution.samples.shape[1] >= N
            sampled_values = self.distribution.samples[self.cov_index,-N:] # retrieve last N samples
            self.value = self.distribution.samples[self.cov_index,-1] # retrieve last sample from distribution
            self.original = self.distribution.samples[self.cov_index,-1] # retrieve last sample from distribution
        else:
            sampled_values = self.original * np.ones(N)

        self.values = sampled_values
        return sampled_values # return the requested number of samples

class Behaviour():
    def __init__(self,key:str=''):
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

class Performance(VisualizeDist):
    def __init__(self,key:str='',type='less_is_better'):
        """
        Contains all the necessary tools to calculate performance
        and store its values if there is stochasticity

        Parameters
        ----------
        key : str, optional
            string to tag instance with, default = ''
        type : str, optional
            specifies the sign of the performance 
            parameter when calculating the impact on it,
            possible values: ('less_is_better','more_is_better'), 
            if more_is_better is selected then the sign is negative,
            of less_is_better is selected then the sign is positive, by default = 'less_is_better'
        """

        self.key                = key
        self.type               = type
        self._values            = np.empty(0)
        self._value_dist        = None
        self.value              = None

        assert self.type in ['less_is_better','more_is_better']

    @property
    def values(self) -> np.ndarray:
        """
        Excess vector getter

        Returns
        -------
        np.1darray
            vector of observations
        """
        return self._values

    @values.setter
    def values(self,v:Union[float,np.ndarray]):
        """
        Appends observation v to values vector

        Parameters
        ----------
        v : Union[float,np.ndarray]
            value to append to response vector
        """
        self._values = np.append(self._values,v)

    @property
    def value_dist(self) -> Distribution:
        """
        Value Distribution object

        Returns
        -------
        dmLib.Distribution
            instance of dmLib.Distribution holding value pdf
        """
        return self._value_dist

    @value_dist.setter
    def value_dist(self,values:np.ndarray):
        """
        Creates value Distribution object

        Parameters
        ----------
        values : np.1darray
            Vector of values
        """
        self._value_dist = Distribution(values, lb=min(values),ub=max(values))

    def view(self,xlabel:str=None,folder:str='',file:str=None,img_format:str='pdf'):
        """
        Views the distribution of the performance parameter

        Parameters
        ----------
        xlabel : str, optional
            axis label of value , if not provided uses the key of the object, 
            by default None
        folder : str, optional
            folder in which to store image, by default ''
        file : str, optional
            name of image file, if not provide then an image is not saved, by default None
        img_format : str, optional
            format of the image to be stored, by default 'pdf'
        """
        if xlabel is None:
            xlabel = '%s' %self.key
        super().view(xlabel=xlabel,folder=self.key,file=file,img_format=img_format)

    def view_cdf(self,xlabel:str=None,folder:str='',file:str=None,img_format:str='pdf'):
        """
        Views the cumulative distribution of the performance parameter

        Parameters
        ----------
        xlabel : str, optional
            axis label of value , if not provided uses the key of the object, 
            by default None
        folder : str, optional
            folder in which to store image, by default ''
        file : str, optional
            name of image file, if not provide then an image is not saved, by default None
        img_format : str, optional
            format of the image to be stored, by default 'pdf'
        """
        if xlabel is None:
            xlabel = '%s' %self.key
        super().view_cdf(xlabel=xlabel,folder=self.key,file=file,img_format=img_format)

    def reset(self,N:int=None):
        """
        Resets accumulated random observations in performance, and value distributions

        Parameters
        -----------
        N : int, optional
            if provided deletes only the last n_samples, by default None
        """

        if N is not None:
            assert N <= len(self._values)
            self._values    = self._values[:-N].copy()
        else:
            self._values    = np.empty(0)

        self._value_dist    = None

    def __call__(self,performance:np.ndarray):
        """
        Set the value of the performance parameter

        Parameters
        ----------
        performance : np.1darray
            values of the performance parameter.
            The length of this vector equals the number of samples
        """

        self.value  = performance # store performance
        self.values = performance # add to list of performance samples

class MarginNode(Performance):
    def __init__(self,key:str='',cutoff:float=0.9,buffer_limit:float=0.0,type:str='must_exceed'):
        """
        Contains description and implementation 
        of a Margin Node object which is the building block
        of a Margin Analysis Network (MAN)

        Parameters
        ----------
        key : str, optional
            string to tag instance with, default = ''
        cutoff : float, optional
            cutoff limit for calculating reliability,
            default = 0.9
        buffer_limit : float, optional
            lower bound for beginning of buffer zone,
            default = 0.0
        type : str, optional
            possible values('must_exceed','must_not_exceed'), by default 'must_exceed'
        """
        Performance.__init__(self,key)
        # no need to call initializer of VisualizeDist (self.values are already defined by this class)
        self.cutoff             = cutoff
        self.buffer_limit       = buffer_limit
        self.type               = type
        self._targets           = np.empty(0)
        self._decided_values    = np.empty(0)
        self.target             = None
        self.decided_value      = None

        assert self.type in ['must_exceed','must_not_exceed']

    @property
    def targets(self) -> np.ndarray:
        """
        Target vector getter

        Returns
        -------
        np.1darray
            vector of target observations
        """
        return self._targets

    @targets.setter
    def targets(self,t:Union[float,np.ndarray]):
        """
        Appends target observation t to target vector

        Parameters
        ----------
        t : Union[float,np.ndarray]
            value to append to target vector
        """
        self._targets = np.append(self._targets,t)

    @property
    def decided_values(self) -> np.ndarray:
        """
        Response vector getter

        Returns
        -------
        np.1darray
            vector of response observations
        """
        return self._decided_values

    @decided_values.setter
    def decided_values(self,r:Union[float,np.ndarray]):
        """
        Appends response observation r to target vector

        Parameters
        ----------
        r : Union[float,np.ndarray]
            value to append to response vector
        """
        self._decided_values = np.append(self._decided_values,r)

    def reset(self,N:int=None):
        """
        Resets accumulated random observations in target, 
        response, and excess attributes

        Parameters
        -----------
        N : int, optional
            if provided deletes only the last n_samples, by default None
        """
        super().reset(N)

        if N is not None:
            self._targets           = self._targets[:-N].copy()
            self._decided_values    = self._decided_values[:-N].copy()
        else:
            self._targets           = np.empty(0)
            self._decided_values    = np.empty(0)

    def __call__(self,target_threshold:np.ndarray,decided_value:np.ndarray):
        """
        Calculate excess given the target threshold and decided value

        Parameters
        ----------
        target_threshold : np.1darray
            target thresholds to the margin node describing the capability of the design.
            The length of this vector equals the number of samples
        decided_value : np.1darray
            The decided values that the design needs to achieve
            The length of this vector equals the number of samples
        """
            
        self.decided_values     = decided_value     # add to list of decided values
        self.targets            = target_threshold  # add to list of targets
        self.decided_value      = decided_value     # store decided value
        self.target             = target_threshold  # store target

        if self.type == 'must_exceed':
            e = target_threshold - decided_value
        elif self.type == 'must_not_exceed':
            e = decided_value - target_threshold
        else:
            raise Exception('Wrong margin type (%s) specified. Possible values are "must_Exceed", "must_not_exceed".' %(str(self.type)))

        self.value             = e                 # store excess
        self.values            = e                 # add to list of excesses

class ImpactMatrix(VisualizeDist):
    def __init__(self,n_margins:int,n_performances:int):
        """
        Stores observations of impact matrix. This class is an attribute of the MarginNetwork class, 
        ImpactMatrix is instantiated by the MarginNetwork class during its initialization

        Parameters
        ----------
        n_margins : int
            number of margin nodes
        n_performances : int
            number of performance parameters
        """
        # no need to call initializer of VisualizeDist (self.values are already defined by this class)
        self.n_margins      = n_margins
        self.n_performances = n_performances
        self.impact         = None
        self._impacts       = np.empty((self.n_margins,self.n_performances,0))

        #_impacts [len(margin_nodes), len(performances), n_samples]

    @property
    def impacts(self) -> np.ndarray:
        """
        Impact 3D matrix getter

        Returns
        -------
        np.ndarray
            vector of impact matrix observations
        """
        return self._impacts

    @impacts.setter
    def impacts(self,i:np.ndarray):
        """
        Appends observation matrix i to impacts 3D matrix

        Parameters
        ----------
        i : ndarray
            value to append to impacts 3D matrix
        """
        assert i.shape == self._impacts.shape[:2]
        self._impacts = np.dstack((self._impacts,i))

    def reset(self):
        """
        Resets accumulated random observations in response, and value distributions, and impact matrix
        """

        self._impacts   = np.empty((self.n_margins,self.n_performances,0))

        #_impacts [len(margin_nodes), len(performances), n_samples]   
    
    def view(self,i_margin:int,i_performance:int,xlabel:str=None,folder:str='',file:str=None,img_format:str='pdf'):
        """
        Views the distribution of the desired margin/performance impact pair

        Parameters
        ----------
        i_margin : int
            index of the margin node (row of impact matrix)
        i_performance : int
            index of the performance parameter (column of impact matrix)
        xlabel : str, optional
            the x-axis label to display on the plot, if not provided simply prints 
            `(E<index>,P<index>)` on the x-axis label, by default None
        savefile : str, optional
            if provided saves a screenshot of the figure to file in pdf format, by default None
        """
        if xlabel is None:
            xlabel = 'IoP (E%i,P%i)' %(i_margin+1,i_performance+1)

        super().__init__(self.impacts[i_margin,i_performance,:]) # instantiate the self.values attribute
        super().view(xlabel=xlabel,folder=folder,file=file,img_format=img_format)

    def view_cdf(self,i_margin:int,i_performance:int,xlabel:str=None,folder:str='',file:str=None,img_format:str='pdf'):
        """
        Views the cumulative distribution of the desired margin/performance impact pair

        Parameters
        ----------
        i_margin : int
            index of the margin node (row of impact matrix)
        i_performance : int
            index of the performance parameter (column of impact matrix)
        xlabel : str, optional
            the x-axis label to display on the plot, if not provided simply prints 
            `(E<index>,P<index>)` on the x-axis label, by default None
        savefile : str, optional
            if provided saves a screenshot of the figure to file in pdf format, by default None
        """
        if xlabel is None:
            xlabel = 'IoP (E%i,P%i)' %(i_margin+1,i_performance+1)

        super().__init__(self.impacts[i_margin,i_performance,:]) # instantiate the self.values attribute
        super().view_cdf(xlabel=xlabel,folder=folder,file=file,img_format=img_format)

    def __call__(self,impact:np.ndarray):
        """
        Append impact matrix to stored impacts and store as current value

        Parameters
        ----------
        impact : np.ndarray
            An observation of the impact matrix
            The size of this vector should be (n_margins,n_performances)
        """

        assert impact.shape == self._impacts.shape[:2]
        self.impact     = impact # store impact matrix
        self.impacts    = impact # add to list of impact matrix samples

class AbsorptionMatrix(VisualizeDist):
    def __init__(self,n_margins:int,n_specs:int):
        """
        Stores observations of absorption and deterioration matrices. This class is an attribute of the MarginNetwork class, 
        AbsorptionMatrix is instantiated by the MarginNetwork class during its initialization

        Parameters
        ----------
        n_margins : int
            number of margin nodes
        n_specs : int
            number of input specifications
        """
        # no need to call initializer of VisualizeDist (self.values are already defined by this class)
        self.n_margins          = n_margins
        self.n_specs            = n_specs
        self.absorption         = None
        self.deterioration      = None

        self._absorptions       = np.empty((self.n_margins,self.n_specs,0))

        #_absorptions [len(margin_nodes), len(inputs_specs), n_samples]

        self._deteriorations    = np.empty((self.n_specs,0))

        #_deteriorations [len(inputs_specs), n_samples]

    @property
    def absorptions(self) -> np.ndarray:
        """
        Absorption 3D matrix getter

        Returns
        -------
        np.ndarray
            vector of absorption matrix observations
        """
        return self._absorptions

    @absorptions.setter
    def absorptions(self,a:np.ndarray):
        """
        Appends absorption matrix a to absorptions 3D matrix

        Parameters
        ----------
        a : ndarray
            value to append to impacts 3D matrix
        """
        assert a.shape == self._absorptions.shape[:2]
        self._absorptions = np.dstack((self._absorptions,a))

    @property
    def deteriorations(self) -> np.ndarray:
        """
        Deterioration 2D matrix getter

        Returns
        -------
        np.ndarray
            vector of deterioration matrix observations
        """
        return self._deteriorations

    @deteriorations.setter
    def deteriorations(self,d:np.ndarray):
        """
        Appends deterioration vector d to deteriorations 2D matrix

        Parameters
        ----------
        d : ndarray
            value to append to deteriorations 2D matrix
        """

        if d.ndim == 1:
            assert len(d) == self.n_specs
            d = d.reshape((self.n_specs,1)) # reshape 1D arrays to 2D
        elif d.ndim == 2:
            assert d.shape[0] == self.n_specs

        self._deteriorations = np.hstack((self._deteriorations,d))

    def reset(self):
        """
        Resets accumulated random observations in absorption, and deterioration
        """

        self._absorptions       = np.empty((self.n_margins,self.n_specs,0))

        #_absorptions [len(margin_nodes), len(inputs_specs), n_samples]

        self._deteriorations    = np.empty((self.n_specs,0))

        #_deteriorations [len(inputs_specs), n_samples]
    
    def view(self,i_margin:int,i_input_spec:int,xlabel:str=None,folder:str='',file:str=None,img_format:str='pdf'):
        """
        Views the distribution of the desired margin/input_spec absorption pair

        Parameters
        ----------
        i_margin : int
            index of the margin node (row of absorption matrix)
        i_input_spec : int
            index of the input specification (column of absorption matrix)
        xlabel : str, optional
            the x-axis label to display on the plot, if not provided simply prints 
            `(E<index>,S<index>)` on the x-axis label, by default None
        savefile : str, optional
            if provided saves a screenshot of the figure to file in pdf format, by default None
        """
        if xlabel is None:
            xlabel = 'CAC (E%i,S%i)' %(i_margin+1,i_input_spec+1)

        super().__init__(self.absorptions[i_margin,i_input_spec,:]) # instantiate the self.values attribute
        super().view(xlabel=xlabel,folder=folder,file=file,img_format=img_format)

    def view_cdf(self,i_margin:int,i_input_spec:int,xlabel:str=None,folder:str='',file:str=None,img_format:str='pdf'):
        """
        Views the cumulative distribution of the desired margin/input_spec absorption pair

        Parameters
        ----------
        i_margin : int
            index of the margin node (row of absorption matrix)
        i_input_spec : int
            index of the input specification (column of absorption matrix)
        xlabel : str, optional
            the x-axis label to display on the plot, if not provided simply prints 
            `(E<index>,S<index>)` on the x-axis label, by default None
        savefile : str, optional
            if provided saves a screenshot of the figure to file in pdf format, by default None
        """
        if xlabel is None:
            xlabel = 'CAC (E%i,C%i)' %(i_margin+1,i_input_spec+1)

        super().__init__(self.absorptions[i_margin,i_input_spec,:]) # instantiate the self.values attribute
        super().view_cdf(xlabel=xlabel,folder=folder,file=file,img_format=img_format)

    def view_det(self,i_input_spec:int,xlabel:str=None,folder:str='',file:str=None,img_format:str='pdf'):
        """
        Views the distribution of the desired deterioration

        Parameters
        ----------
        i_input_spec : int
            index of the input specification (entry of deterioration vector)
        xlabel : str, optional
            the x-axis label to display on the plot, if not provided simply prints 
            `(S<index>)` on the x-axis label, by default None
        savefile : str, optional
            if provided saves a screenshot of the figure to file in pdf format, by default None
        """
        if xlabel is None:
            xlabel = 'deterioration (S%i)' %(i_input_spec+1)

        super().__init__(self.deteriorations[i_input_spec,:]) # instantiate the self.values attribute
        super().view(xlabel=xlabel,folder=folder,file=file,img_format=img_format)

    def view_det_cdf(self,i_input_spec:int,xlabel:str=None,folder:str='',file:str=None,img_format:str='pdf'):
        """
        Views the cumulative distribution of the desired deterioration

        Parameters
        ----------
        i_input_spec : int
            index of the input specification (column of deterioration vector)
        xlabel : str, optional
            the x-axis label to display on the plot, if not provided simply prints 
            `(S<index>)` on the x-axis label, by default None
        savefile : str, optional
            if provided saves a screenshot of the figure to file in pdf format, by default None
        """
        if xlabel is None:
            xlabel = 'deterioration (S%i)' %(i_input_spec+1)

        super().__init__(self.deteriorations[i_input_spec,:]) # instantiate the self.values attribute
        super().view_cdf(xlabel=xlabel,folder=folder,file=file,img_format=img_format)

    def __call__(self,deterioration:np.ndarray,absorption:np.ndarray):
        """
        Append impact matrix to stored impacts and store as current value

        Parameters
        ----------
        deterioration : np.ndarray
            An observation of the deterioration vector
            The size of this vector should be (n_performances)
        absorption : np.ndarray
            An observation of the absorption matrix
            The size of this matrix should be (n_margins,n_input_specs)
        """

        assert absorption.shape == self._absorptions.shape[:2]
        assert deterioration.ndim == 1 and len(deterioration) == self._deteriorations.shape[0]
        self.deterioration  = deterioration # store deterioration vector
        self.deteriorations = deterioration # add to deterioration vector samples
        self.absorption     = absorption # store absorption matrix
        self.absorptions    = absorption # add to list of absorption matrix samples

class MarginNetwork():
    def __init__(self,design_params:List[DesignParam],input_specs:List[InputSpec],
        fixed_params:List[FixedParam],behaviours:List[Behaviour],
        margin_nodes:List[MarginNode],performances:List[Performance],key:str=''):
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
         margin_nodes : List[Performance]
            list of Performance instances
        key : str, optional
            string to tag instance with, default = ''
        """
        self.key                = key
        
        # Inputs
        self.design_params      = design_params
        self.input_specs        = input_specs
        self.fixed_params       = fixed_params
        self.behaviours         = behaviours

        # Outputs
        self.margin_nodes       = margin_nodes
        self.performances       = performances
        self.impact_matrix      = ImpactMatrix(len(margin_nodes),len(performances))
        self.absorption_matrix  = AbsorptionMatrix(len(margin_nodes),len(input_specs))

        # Design parameter space
        lb = np.array([])
        ub = np.array([])
        # Get upper and lower bound for continuous variables
        for design_param in self.design_params:
            lb = np.append(lb,design_param.universe[0])
            ub = np.append(ub,design_param.universe[1])
        self.lb_d, self.ub_d = lb, ub

    @property
    def design_vector(self) -> np.ndarray:
        """
        returns a vector of design parameters

        Returns
        -------
        np.ndarray
            vector of design parameters
        """
        vector = np.empty(0)
        for item in self.design_params:
            vector = np.append(vector,item.value)
        return vector

    @design_vector.setter
    def design_vector(self,d:np.ndarray):
        """
        Adjusts all the design parameters according to the vector d provided

        Parameters
        ----------
        d : ndarray
            value to set the input specs of the MAN to
        """
        assert d.ndim == 1
        assert len(d) == len(self.design_params)

        for value,item in zip(d,self.design_params):
            item.value = value

    @property
    def spec_vector(self) -> np.ndarray:
        """
        returns a vector of input specifications

        Returns
        -------
        np.ndarray
            vector of input specifications
        """
        vector = np.empty(0)
        for item in self.input_specs:
            vector = np.append(vector,item.value)
        return vector

    @spec_vector.setter
    def spec_vector(self,d:np.ndarray):
        """
        Adjusts all the input specs according to the  vector d provided

        Parameters
        ----------
        d : ndarray
            value to set the input specs of the MAN to
        """
        assert d.ndim == 1
        assert len(d) == self.input_specs

        for value,item in zip(d,self.input_specs):
            item.value = value

    @property
    def nominal_spec_vector(self) -> np.ndarray:
        """
        returns a vector of nominal input specifications

        Returns
        -------
        np.ndarray
            vector of nominal input specifications
        """
        vector = np.empty(0)
        for item in self.input_specs:
            vector = np.append(vector,item.original)
        return vector

    @property
    def excess_vector(self) -> np.ndarray:
        """
        excess vector getter

        Returns
        -------
        np.ndarray
            vector of excesses
        """
        vector = np.empty(0)
        for item in self.margin_nodes:
            vector = np.append(vector,item.value)
        return vector

    @property
    def dv_vector(self) -> np.ndarray:
        """
        decided value vector getter

        Returns
        -------
        np.ndarray
            vector of decided values
        """
        vector = np.empty(0)
        for item in self.margin_nodes:
            vector = np.append(vector,item.decided_value)
        return vector

    @property
    def tt_vector(self) -> np.ndarray:
        """
        target threshold vector getter

        Returns
        -------
        np.ndarray
            vector of target thresholds
        """
        vector = np.empty(0)
        for item in self.margin_nodes:
            vector = np.append(vector,item.target)
        return vector

    @property
    def perf_vector(self) -> np.ndarray:
        """
        performance parameter vector getter

        Returns
        -------
        np.ndarray
            vector of performance parameters
        """
        vector = np.empty(0)
        for item in self.performances:
            vector = np.append(vector,item.value)
        return vector

    def train_performance_surrogate(self,n_samples:int=100,sampling_freq:int=1,ext_samples:Tuple[np.ndarray]=None):
        """
        Constructs a surrogate model y(x), where x are the excess values and 
        y are the performance parameters that can be used to calculate threshold 
        performances

        Parameters
        ----------
        n_samples : int, optional
            number of design space data points used for training, by default 100
        sampling_freq : int, optional
            If > than 1 then the decided value is calculated as the average of N samples 
            by calling the forward() N times and averaging the decided values, where N = sampling_freq, by default 1
        ext_samples : tuple[np.ndarray], optional
            if sample data provided externally then use directly to fit the response surface, 
            Tuple must of length 2, ext_samples[0] must of shape (N_samples,len(margin_nodes)),
            ext_samples[1] must of shape (N_samples,len(performances)),
            by default None
        """

        if ext_samples is None:
            # generate training data for response surface using a LHS grid of design parameter space
            design_samples = Design(self.lb_d,self.ub_d,n_samples,"LHS").unscale() # 2D grid

            xt = np.empty((0,len(self.margin_nodes))) # decided values
            yt = np.empty((0,len(self.performances))) # Performance parameters

            for design in design_samples:

                self.design_vector = design # Set design parameters to their respective values
                
                excess_samples = np.empty((0,len(self.margin_nodes)))
                perf_samples = np.empty((0,len(self.performances)))
                for n in range(sampling_freq):
                    self.randomize() # Randomize the MAN
                    self.forward() # Run one pass of the MAN

                    excess_samples = np.vstack((excess_samples,self.excess_vector.reshape(1,excess_samples.shape[1])))
                    perf_samples = np.vstack((perf_samples,self.perf_vector.reshape(1,perf_samples.shape[1])))

                excess_samples = np.mean(excess_samples,axis=0)
                perf_samples = np.mean(perf_samples,axis=0)

                xt = np.vstack((xt,excess_samples.reshape(1,xt.shape[1])))
                yt = np.vstack((yt,perf_samples.reshape(1,yt.shape[1])))

        elif ext_samples is not None:
            assert type(ext_samples) == tuple
            assert len(ext_samples) == 2
            for value in ext_samples:
                assert type(value) == np.ndarray
            assert ext_samples[0].shape[1] == len(self.margin_nodes)
            assert ext_samples[1].shape[1] == len(self.performances)
            assert ext_samples[0].shape[0] == ext_samples[1].shape[0]

            xt = ext_samples[0]
            yt = ext_samples[1]

        # Get lower and upper bounds of excess values
        self.lb_excess = np.min(xt,axis=0)
        self.ub_excess = np.max(xt,axis=0)

        self.reset()
        self.sm_perf = KRG(theta0=[1e-2])
        self.sm_perf.set_training_values(xt, yt)
        self.sm_perf.train()

    def view_perf(self,e_indices:List[int],p_index:int,label_1:str=None,label_2:str=None,
        label_p:str=None,n_levels:int=100,folder:str='',file:str=None,img_format:str='pdf'):
        """
        Shows the estimated performance 

        Parameters
        ----------
        e_indices : list[int]
            index of the excess values to be viewed on the plot
        p_index : int
            index of the performance parameter to be viewed on the plot
        label_1 : str, optional
            axis label of excess value 1if not provided uses the key of MarginNode, 
            by default None
        label_2 : str, optional
            axis label of excess value 2, if not provided uses the key of MarginNode, 
            by default None
        label_2 : str, optional
            z-axis label of performance parameter, if not provided uses the key of Performance object, 
            by default None
        n_levels : int, optional
            resolution of the plot (how many full factorial levels in each direction are sampled), 
            by default 100
        folder : str, optional
            folder in which to store image, by default ''
        file : str, optional
            name of image file, if not provide then an image is not saved, by default None
        img_format : str, optional
            format of the image to be stored, by default 'pdf'
        """

        # Plot the result in 2D
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot()

        sampling_vector = np.ones(len(self.margin_nodes))
        # only sample the selected variables while holding the other variables at their nominal values
        sampling_vector[e_indices] = n_levels

        lb_excess, ub_excess = np.empty(0), np.empty(0)
        for margin_node in self.margin_nodes:
            lb_excess = np.append(lb_excess,margin_node.value)
            ub_excess = np.append(ub_excess,margin_node.value+1e-3)
        lb_excess[e_indices] = self.lb_excess[e_indices]
        ub_excess[e_indices] = self.ub_excess[e_indices]

        excess_space = Design(lb_excess,ub_excess,sampling_vector,'fullfact').unscale()
        perf_estimate = self.sm_perf.predict_values(excess_space)

        x = excess_space[:,e_indices[0]].reshape((n_levels,n_levels))
        y = excess_space[:,e_indices[1]].reshape((n_levels,n_levels))
        z = perf_estimate[:,p_index].reshape((n_levels,n_levels))

        if label_1 is None:
            label_1 = 'E_%s' %self.margin_nodes[e_indices[0]].key
        if label_2 is None:
            label_2 = 'E_%s' %self.margin_nodes[e_indices[1]].key
        if label_p is None:
            label_p = self.performances[p_index].key

        surf = ax.contourf(x, y, z, cmap=plt.cm.jet,)
        ax.set_xlabel(label_1)
        ax.set_ylabel(label_2)

        cbar = plt.cm.ScalarMappable(cmap=plt.cm.jet)
        cbar.set_array(z)

        boundaries = np.linspace(np.min(perf_estimate[:,p_index]), np.max(perf_estimate[:,p_index]), 51)
        cbar_h = fig.colorbar(cbar, boundaries=boundaries)
        cbar_h.set_label(label_p, rotation=90, labelpad=3)

        if file is not None:
            # Save figure to image
            check_folder('images/%s' %(folder))
            self.fig.savefig('images/%s/%s.%s' %(folder,file,img_format), 
                format=img_format, dpi=200, bbox_inches='tight')

        plt.show()

    def compute_impact(self,use_estimate:bool=True):
        """
        Computes the impact on performance (IoP) matrix of the MVM that has a size
        [n_margins, n_performances] and appends its the impacts matrix if called 
        multiple times without a reset(). 
        Must be called after executing at least one `forward()` pass

        Parameters
        ----------
        use_estimate : bool, optional
            uses the trained surrogate model from `train_performance_surrogate` 
            to estimate performance at the threshold design, by default True
        """

        excess_values = self.excess_vector.reshape(1,-1) # turn it into a 2D matrix

        # Compute performances at decided values first
        if use_estimate:
            # Use surrogate model
            performances = self.sm_perf.predict_values(excess_values)
        else:
            # Get performances from behaviour models
            performances = self.perf_vector

        performances = np.tile(performances,(len(self.margin_nodes),1))

        #performances = [len(margin_nodes), len(performances)]

        # Compute performances at target threshold for each margin node
        input = np.tile(excess_values,(len(self.margin_nodes),1)) # Create a square matrix

        #input = [len(margin_nodes), len(margin_nodes)]

        np.fill_diagonal(input, np.zeros(len(self.margin_nodes)), wrap=False) # works in place

        #input = [len(margin_nodes), len(margin_nodes)]

        thresh_perf = self.sm_perf.predict_values(input)

        #thresh_perf = [len(margin_nodes), len(performances)]

        # Get the sign of the performance parameters
        signs = np.empty(0)
        for performance in self.performances:
            if performance.type == 'less_is_better':
                sign = 1.0
            elif performance.type == 'more_is_better':
                sign = -1.0
            signs = np.append(signs,sign)

        impact = signs * (performances - thresh_perf) / thresh_perf

        #impact = [len(margin_nodes), len(performances)]

        self.impact_matrix(impact) # Store impact matrix

    def compute_absorption(self):
        """
        Computes the change absorption capability (CAC) matrix of the MVM that has a size
        [n_margins, n_input_specs] and appends its the absorption matrix if called 
        multiple times without a reset(). 
        Also stores and appends the deterioration vectors
        """
        
        # Deterioration computation

        spec_limit = np.empty(0)
        for spec in self.input_specs:

            self.forward() # do not randomize the man for deterioration
            n_inc = 1
            delta_E = 1.0

            if spec.inc_type == 'rel':
                inc = (spec.original * spec.inc) / 100
            elif spec.inc_type == 'abs':
                inc = spec.inc

            while all(self.excess_vector >= 0) and delta_E <= 1e3 and abs(delta_E) >= 1e-3:
                
                excess_last_inc = self.excess_vector

                spec.value += inc
                self.forward() # do not randomize the man for deterioration
                n_inc += 1

                delta_E = np.min(self.excess_vector - excess_last_inc)

            spec_limit = np.append(spec_limit,spec.value + inc)

            self.reset(n_inc)

        deterioration = (spec_limit - self.nominal_spec_vector) / self.nominal_spec_vector

        # Absorption computation

        nominal_threshold = np.reshape(self.tt_vector,(len(self.margin_nodes),-1))
        target_thresholds = np.tile(nominal_threshold,(1,len(self.input_specs)))

        #target_thresholds = [len(margin_nodes), len(input_specs)]

        deterioration_matrix = np.tile(deterioration,(len(self.margin_nodes),1))

        #target_thresholds = [len(margin_nodes), len(input_specs)]

        # Compute performances at the spec limit for each margin node

        change_matrix = np.empty((len(self.margin_nodes),0))
        for input_spec,input_spec_limit in zip(self.input_specs,spec_limit):
            input_spec.value = input_spec_limit
            self.forward()

            change_vector = np.reshape(self.tt_vector,(len(self.margin_nodes),-1))
            change_matrix = np.hstack((change_matrix,change_vector))

            self.reset(1)

        absorption = (change_matrix - target_thresholds) / (target_thresholds * deterioration_matrix)

        #absorption = [len(margin_nodes), len(input_specs)]

        self.absorption_matrix(deterioration=deterioration,absorption=absorption) # store the results

    def randomize(self, *args, **kwargs):
        """
        The function that will be used to randomize the MAN. 
        Include all random distribution objects here and have 
        them draw samples by calling them. See example below

        Example
        -------
        >>> # [in the plugin file]
        >>> from dmLib import guassianFunc, MarginNetwork, InputSpec
        >>> dist_1 = guassianFunc(1.0,0.1)
        >>> dist_2 = guassianFunc(0.5,0.2)
        >>> s1 = InputSpec(1.0 ,'S1', distribution=dist_1)
        >>> s2 = InputSpec(0.5 ,'S2', distribution=dist_1)
        >>> class myMarginNetwork(MarginNetwork):
        >>>     def randomize(self):
        >>>         # randomization procedure
        >>>         Requirement_1()
        >>>         Requirement_2()
        >>>     def forward(self):
        >>>         # some specific model-dependent behaviour
        >>>         x = s1.value + s2.value # These values will be drawn from their respective distributions
        """
        # default code for the default threshold
        pass

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
        pass

    def reset(self,N:int=None):
        """
        Resets all elements of the MAN by clearing their internal caches

        Parameters
        -----------
        N : int, optional
            if provided deletes only the last n_samples, where possible, by default None
        """

        for design_param in self.design_params:
            design_param.reset()
        for input_spec in self.input_specs:
            input_spec.reset()
        for behaviour in self.behaviours:
            behaviour.reset()
        for margin_node in self.margin_nodes:
            margin_node.reset(N)
        for performance in self.performances:
            performance.reset(N)

    def reset_outputs(self):
        """
        Resets Impact, Absorption, and Deterioration matrices
        """

        self.impact_matrix.reset()
        self.absorption_matrix.reset()