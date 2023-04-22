from abc import ABC, abstractmethod
from typing import Tuple, List, Union, Callable, Dict
from copy import copy,deepcopy
import multiprocess as mp
import sys
import os


import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.stats as st
from smt.surrogate_models import KRG, LS, QP
from smt.sampling_methods import LHS
from smt.applications.mixed_integer import (
    FLOAT,
    ORD,
    ENUM,
    MixedIntegerSamplingMethod,
    MixedIntegerSurrogateModel
)
from smt.applications import MOE

from .DOELib import Design, scaling
from .uncertaintyLib import Distribution, GaussianFunc, UniformFunc, VisualizeDist
from .utilities import check_folder, parallel_sampling, clamp

"""Design margins library for computing buffer and excess"""

class Cache(ABC):

    def __init__(self, key: str, dims: List[int]):
        """
        Stores observations data. This class is an attribute of the MarginNetwork class, 
        Cache subclasses are instantiated by the MarginNetwork class during its initialization

        Parameters
        ----------
        key : str
            unique identifier
        dims : List[int]
            The dimension along each axis (if empty then a float is assumed)
        """
        self.key = key
        self.dims = dims

        self.value = None
        self._values = np.empty(self.dims + [0, ])
        self.ndim = self._values.ndim

    @property
    @abstractmethod
    def values(self):
        pass

    @values.setter
    @abstractmethod
    def values(self, value):
        pass

    @abstractmethod
    def view(self, *args, **kwargs):
        pass

    @abstractmethod
    def view_cdf(self, *args, **kwargs):
        pass

    def reset(self, n: int):
        """
        Resets accumulated random observations and value distributions

        Parameters
        -----------
        n : int, optional
            if provided deletes only the last n_samples, by default None
        """
        if n is not None:
            assert n <= len(self._values), 'n must be a less than the number of samples of the scalar, expected a number less than %i and got %i' %(len(self._values),n)
            if n > 0:
                if n == len(self._values):
                    self._values = np.empty(self.dims + [0, ])
                    self.value = None
                else:
                    self._values = self._values[..., :-n].copy()  # select along last dimension
                    self.value = self._values[..., -1].copy()
            else:
                pass # do not reset anything

        else:
            self._values = np.empty(self.dims + [0, ])
            self.value = None

    def __call__(self, value: Union[float, np.ndarray]):
        """
        Set the value of the parameter

        Parameters
        ----------
        value : Union[float,np.ndarray]
            values of the parameter.
            The length of this vector equals the number of samples
        """

        self.value = value  # store value
        self.values = value  # add to list of samples

    def __deepcopy__(self, memo, datatype: str = None): # memo is a dict of id's to copies
        """
        creates a deep independent copy of the class instance self.
        https://stackoverflow.com/a/15774013

        Parameters
        ----------
        memo : Dict
            memoization dictionary of id(original) (or identity numbers) to copies
        datatype : str, optional
            The type of values being stored in ScalarParams, 
            possible values : {'continuous', 'categorical'}, by default None

        Returns
        -------
        Decision
            copy of Cache instance
        """
        id_self = id(self) # memoization avoids unnecessary recursion
        _copy = memo.get(id_self)
        if _copy is None:
            if datatype is not None:
                _copy = type(self)(self.key, deepcopy(self.dims), datatype)
            else:
                _copy = type(self)(self.key, deepcopy(self.dims))
            _copy._values = self._values
            _copy.value = self.value
            memo[id_self] = _copy 
        return _copy


class ScalarParam(Cache):
    def __init__(self, key: str, dims: List[int], datatype: str = 'continuous'):
        """
        Stores observations data. This class is an attribute of the MarginNetwork class, 
        Cache subclasses are instantiated by the MarginNetwork class during its initialization

        Parameters
        ----------
        key : str
            unique identifier
        dims : List[int]
            The dimension along each axis (if empty then a float is assumed)
        datatype : str, optional
            The type of values being stored, 
            possible values : {'continuous', 'categorical'}, by default 'categorical'
        """
        super().__init__(key, dims)

        assert datatype in ['continuous','categorical'], \
            'expected datatype in "continuous", "categorical", got %d' %datatype

        self.datatype = datatype
        self._value_dist = None

    @property
    def values(self) -> np.ndarray:
        """
        value vector getter

        Returns
        -------
        np.ndarray
            vector of observations
        """
        return self._values

    @values.setter
    def values(self, v: Union[float, np.ndarray]):
        """
        Appends observation v to values vector

        Parameters
        ----------
        v : Union[float,np.ndarray]
            value to append to response vector
        """
        if type(v) == np.ndarray:
            assert v.ndim == 1

        self._values = np.append(self._values, v)

    @property
    def value_dist(self) -> Distribution:
        """
        Value Distribution object

        Returns
        -------
        Distribution
            instance of Distribution holding value pdf
        """
        return self._value_dist

    @value_dist.setter
    def value_dist(self, values: np.ndarray):
        """
        Creates value Distribution object

        Parameters
        ----------
        values : np.ndarray
            Vector of values
        """

        assert self.datatype == 'continuous', 'This method cannot be used with categorical variables'

        try:
            with np.errstate(divide='ignore'):
                value_hist = np.histogram(values, bins=50, density=True)
                self._value_dist = Distribution(value_hist[0], lb=min(value_hist[1]), ub=max(value_hist[1]))
        except:
            self._value_dist = None

    def view(self, xlabel: str = None, folder: str = '', file: str = None, img_format: str = 'pdf',
             bins: List[Union[int,float,str]] = None):
        """
        Views the distribution of the parameter

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
        bins : List[Union[int,float,str]], optional
            bins to show, by default None
        """
        if xlabel is None:
            xlabel = '%s' % self.key
        
        vis = VisualizeDist(values=self.values)

        if self.datatype == 'continuous':
            vis.view(xlabel=xlabel, folder=self.key, file=file, img_format=img_format)
        elif self.datatype == 'categorical':
            vis.view(xlabel=xlabel, folder=self.key, file=file, img_format=img_format,
                bins=bins, bar_width=0.7)

    def view_cdf(self, xlabel: str = None, folder: str = '', file: str = None, img_format: str = 'pdf'):
        """
        Views the cumulative distribution of the parameter

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
        assert self.datatype == 'continuous', 'This method cannot be used with categorical variables'

        if xlabel is None:
            xlabel = '%s' % self.key

        vis = VisualizeDist(values=self.values)
        vis.view_cdf(xlabel=xlabel, folder=self.key, file=file, img_format=img_format)

    def __deepcopy__(self, memo): # memo is a dict of id's to copies
        """
        creates a deep independent copy of the class instance self.
        https://stackoverflow.com/a/15774013

        Parameters
        ----------
        memo : Dict
            memoization dictionary of id(original) (or identity numbers) to copies

        Returns
        -------
        Decision
            copy of ScalarParam instance
        """
        _copy = super().__deepcopy__(memo, self.datatype)
        _copy._value_dist = self._value_dist
        return _copy


class VectorParam(Cache):
    """
    Stores observations of a vector. This class is an attribute of the MarginNetwork class, 
    MatrixParam is instantiated by the MarginNetwork class during its initialization
    """

    @property
    def values(self) -> np.ndarray:
        """
        Impact 3D matrix getter

        Returns
        -------
        np.ndarray
            vector of matrix observations
        """
        return self._values

    @values.setter
    def values(self, v: np.ndarray):
        """
        Appends observation matrix i to 3D matrix

        Parameters
        ----------
        v : ndarray
            value to append to vector
            can be 1 dimensional vector or a 2 dimensional column vector
        """

        assert v.shape[0] == self.dims[0]
        if v.ndim == 1:
            v = v.reshape(self.dims + [1, ])  # reshape 1D arrays to 2D
        self._values = np.hstack((self._values, v))

    def view(self, row: int, xlabel: str = None, folder: str = '', file: str = None, img_format: str = 'pdf'):
        """
        Views the distribution of the desired vector component

        Parameters
        ----------
        row : int
            index of the row of vector
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
            xlabel = 'R%i' % (row + 1,)

        vis = VisualizeDist(values=self.values[row, :])
        vis.view(xlabel=xlabel, folder=folder, file=file, img_format=img_format)

    def view_cdf(self, row: int, xlabel: str = None, folder: str = '', file: str = None, img_format: str = 'pdf'):
        """
        Views the distribution of the desired vector component

        Parameters
        ----------
        row : int
            index of the row of vector
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
            xlabel = 'R%i' % (row + 1)

        vis = VisualizeDist(values=self.values[row, :])
        vis.view_cdf(xlabel=xlabel, folder=folder, file=file, img_format=img_format)


class MatrixParam(Cache):
    """
    Stores observations of a matrix. This class is an attribute of the MarginNetwork class, 
    MatrixParam is instantiated by the MarginNetwork class during its initialization
    """

    @property
    def values(self) -> np.ndarray:
        """
        Impact 3D matrix getter

        Returns
        -------
        np.ndarray
            vector of matrix observations
        """
        return self._values

    @values.setter
    def values(self, v: np.ndarray):
        """
        Appends observation matrix i to 3D matrix

        Parameters
        ----------
        v : ndarray
            value to append to 3D matrix
        """
        assert v.shape == self._values.shape[:self.ndim - 1]
        self._values = np.dstack((self._values, v))

    def view(self, row: int, col: int, xlabel: str = None, folder: str = '', file: str = None, img_format: str = 'pdf'):
        """
        Views the distribution of the desired matrix element

        Parameters
        ----------
        row : int
            index of the row of matrix
        col : int
            index of the column of matrix
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
            xlabel = 'R%i,C%i' % (row + 1, col + 1)

        vis = VisualizeDist(values=self.values[row, col, :])
        vis.view(xlabel=xlabel, folder=folder, file=file, img_format=img_format)

    def view_cdf(self, row: int, col: int, xlabel: str = None, folder: str = '', file: str = None,
                 img_format: str = 'pdf'):
        """
        Views the distribution of the desired matrix element

        Parameters
        ----------
        row : int
            index of the row of matrix
        col : int
            index of the column of matrix
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
            xlabel = 'R%i,C%i' % (row + 1, col + 1)

        vis = VisualizeDist(values=self.values[row, col, :])
        vis.view_cdf(xlabel=xlabel, folder=folder, file=file, img_format=img_format)


class ParamFactory:
    """
    Constructs different parameters based on given dimensions
    """

    @staticmethod
    def build_param(key: str, dims: List[int], datatype: str = 'continuous') -> Union[ScalarParam, VectorParam, MatrixParam]:
        """
        Returns appropriate parameter class based on supplied dimensions

        Parameters
        ----------
        key : str
            unique string identifier
        dims : List[int]
            The dimensions of the parameter
        datatype : str, optional
            The type of values being stored, 
            possible values : {'continuous', 'categorical'}, by default 'categorical'

        Returns
        -------
        Union[ScalarParam,VectorParam,MatrixParam]
            an instance of the correct class
        """
        if len(dims) == 0:
            return ScalarParam(key, dims, datatype)
        elif len(dims) == 1:
            return VectorParam(key, dims)
        elif len(dims) == 2:
            return MatrixParam(key, dims)


class FixedParam:
    def __init__(self, value: Union[float, int, str], key: str,
                 description: str = '', symbol: str = ''):
        """
        Contains description of an input parameter to the MAN

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

        self.description = description
        self.symbol = symbol
        self.key = key
        self.type = type(value)
        self.value = value


class DesignParam:
    def __init__(self, value: Union[float, int, str], key: str,
                 universe: List[Union[int, float]], variable_type: str,
                 description: str = '', symbol: str = ''):
        """
        Contains description of an input parameter to the MAN
        is inherited by DesignParam, and FixedParam

        Parameters
        ----------
        value : Union[float,int,str]
            the value of the input spec
        key : str
            unique identifier
        universe : List[Union[int, float]]
            the possible values the design parameter can take, 
            must be of length 2 (upper and lower bound)
            type(value) must be float, or int
        variable_type : str
            type of variable, possible values are 'INT', 'FLOAT'
        description : str, optional
            description string, by default ''
        symbol : str, optional
            shorthand symbol, by default ''
        """
        self.description = description
        self.symbol = symbol
        self.key = key
        self.type = type(value)
        self.variable_type = variable_type
        self.value = value

        assert len(universe) == 2, 'Universe must have a length of 2, %i given' % len(universe)
        assert self.type in [float, int]
        self.original = value
        self.universe = universe

    def reset(self):
        """
        resets the design parameters to its initial value given at `__init__`   
        """
        self.value = self.original


class InputSpec(ScalarParam):
    def __init__(self, value: Union[float, int],
                 key: str, universe: List[Union[int, float]], variable_type: str,
                 description: str = '', symbol: str = '', 
                 distribution: Union[Distribution, GaussianFunc, UniformFunc] = None,
                 cov_index: int = 0, inc: float = 5.0, inc_type: str = 'rel'):
        """
        Contains description of an input specification
        could deterministic or stochastic

        Parameters
        ----------
        value : Union[float,int]
            the value of the input spec
        key : str
            unique identifier
        universe : List[Union[int, float]]
            the possible values the design parameter can take, 
            must be of length 2 (upper and lower bound)
            type(value) must be float or int
        variable_type : str
            type of variable, possible values are 'INT', 'FLOAT'
        description : str, optional
            description string, by default ''
        symbol : str, optional
            shorthand symbol, by default ''
        distribution : Union[Distribution, GaussianFunc, UniformFunc], optional
            if a Distribution object is provided, then the spec can sampled by calling it
            Example:
            >>> from mvm import InputSpec, GaussianFunc
            >>> dist = GaussianFunc(1.0, 0.1)
            >>> s1 = InputSpec(1.0, 'S1', [0.0, 1.0], 'FLOAT', distribution=dist)
            >>> sample = s1.random()
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

        super().__init__(key=key, dims=[])

        assert type(universe) == list, 'universe must be a list of length 2, got %s' %(universe)

        self.description = description
        self.symbol = symbol
        self.distribution = distribution
        self.cov_index = cov_index
        self.inc_user = inc
        self.inc_type = inc_type
        self.original = value
        self.universe = universe
        self.variable_type = variable_type

        self.value = value

        # Check if input spec is stochastic
        if type(distribution) in [Distribution, GaussianFunc, UniformFunc]:
            self.stochastic = True
        else:
            self.stochastic = False

        assert len(universe) == 2, 'Universe must have a length of 2, %i given' % len(universe)

    @property
    def inc(self) -> float:
        """
        returns the adjusted change increment value for the input specification

        Returns
        -------
        float
            value of adjusted input specification change increments
        """
        if self.inc_type == 'rel':
            _inc = (abs(self.original) * self.inc_user) / 100
        elif self.inc_type == 'abs':
            _inc = self.inc_user
        else:
            _inc = self.inc_user
            Warning('increment type %s, is invalid using absolute value' % str(self.inc_user))

        return _inc

    def reset(self, n: Union[int,str] = None):
        """
        Resets accumulated random observations and value distributions

        Parameters
        -----------
        n : Union[int,str], optional
            if provided deletes only the last n_samples and resets specs to original nominal value, 
            if the string 'all' is provided deletes all samples and resets specs to original nominal value,
            if None is provied resets specs to original nominal value only, by default None
        """
        if n is not None and isinstance(n,int): # only reset if an integer is provided (to avoid clearing samples)
            super().reset(n)
        elif n == 'all':
            super().reset(None)
            
        self.value = self.original

    def random(self, n: int = 1) -> np.ndarray:
        """
        draw random samples from value

        Parameters
        ----------
        n : int, optional
            Number of random samples to draw
            default is one sample

        Returns
        -------
        np.ndarray
            A 1D array of size N, where 
            N is the number of requested samples
        """

        self.distribution.random(n) # draw a sample from distribution
        if self.stochastic:
            assert self.distribution.samples.shape[1] >= n
            sampled_values = self.distribution.samples[self.cov_index, -n:]  # retrieve last N samples
            self.value = self.distribution.samples[self.cov_index, -1]  # retrieve last sample from distribution
            self.original = self.distribution.samples[self.cov_index, -1]  # retrieve last sample from distribution
        else:
            sampled_values = self.original * np.ones(n)

        self.values = sampled_values
        return sampled_values  # return the requested number of samples

    def __deepcopy__(self, memo): # memo is a dict of id's to copies
        """
        creates a deep independent copy of the class instance self.
        https://stackoverflow.com/a/15774013

        Parameters
        ----------
        memo : Dict
            memoization dictionary of id(original) (or identity numbers) to copies

        Returns
        -------
        Decision
            copy of InputSpec instance
        """
        id_self = id(self) # memoization avoids unnecessary recursion
        _copy = memo.get(id_self)
        if _copy is None:

            _copy = type(self)(self.value, self.key, deepcopy(self.universe, memo), self.variable_type,
                               self.description, self.symbol, deepcopy(self.distribution, memo),
                               self.cov_index, self.inc, self.inc_type)
            _copy.key = self.key
            _copy._values = self._values
            _copy.value = self.value
            _copy._value_dist = self._value_dist
            memo[id_self] = _copy 
        return _copy


class Performance(ScalarParam):
    def __init__(self, key: str = '', direction: str = 'less_is_better'):
        """
        Contains all the necessary tools to calculate performance
        and store its values if there is stochasticity

        Parameters
        ----------
        key : str, optional
            string to tag instance with, default = ''
        direction : str, optional
            specifies the sign of the performance 
            parameter when calculating the impact on it,
            possible values: ('less_is_better','more_is_better'), 
            if more_is_better is selected then the sign is negative,
            of less_is_better is selected then the sign is positive, by default = 'less_is_better'
        """

        super().__init__(key=key, dims=[])
        self.direction = direction

        assert self.direction in ['less_is_better', 'more_is_better']
    
    def __deepcopy__(self, memo): # memo is a dict of id's to copies
        """
        creates a deep independent copy of the class instance self.
        https://stackoverflow.com/a/15774013

        Parameters
        ----------
        memo : Dict
            memoization dictionary of id(original) (or identity numbers) to copies

        Returns
        -------
        Decision
            copy of InputSpec instance
        """
        id_self = id(self) # memoization avoids unnecessary recursion
        _copy = memo.get(id_self)
        if _copy is None:

            _copy = type(self)(self.key, self.direction)
            _copy.key = self.key
            memo[id_self] = _copy 
        return _copy


class MarginNode:
    def __init__(self, key: str = '', cutoff: float = 0.9, buffer_limit: float = 0.0, direction: str = 'must_exceed'):
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
        direction : str, optional
            possible values('must_exceed','must_not_exceed'), by default 'must_exceed'
        """

        self.key = key
        self.direction = direction
        self.cutoff = cutoff
        self.buffer_limit = buffer_limit

        self.target = ParamFactory.build_param(key=self.key, dims=[])
        self.decided_value = ParamFactory.build_param(key=self.key, dims=[])
        self.excess = ParamFactory.build_param(key=self.key, dims=[])

        assert self.direction in ['must_exceed', 'must_not_exceed']

    def reset(self, n: int = None):
        """
        Resets accumulated random observations in target, 
        decided value, and excess attributes

        Parameters
        -----------
        n : int, optional
            if provided deletes only the last n_samples, by default None
        """
        self.target.reset(n)
        self.decided_value.reset(n)
        self.excess.reset(n)

    def save(self,filename: str = 'man'):
        """
        saves the MarginNode's stored matrices

        Parameters
        ----------
        filename : str, optional
           basefile path, by default 'man'
        """
        
        with open(filename+'_margin_node_%s.pkl' % self.key,'wb') as f:
            pickle.dump(self.excess,f)
            pickle.dump(self.decided_value,f)
            pickle.dump(self.target,f)

    def load(self,filename='man'):
        """
        loads the MarginNode's stored matrices

        Parameters
        ----------
        filename : str, optional
           basefile path, by default 'man'
        """
        
        with open(filename+'_margin_node_%s.pkl' % self.key,'rb') as f:
            self.excess = pickle.load(f)
            self.decided_value = pickle.load(f)
            self.target = pickle.load(f)

    def __call__(self, target_threshold: np.ndarray, decided_value: np.ndarray):
        """
        Calculate excess given the target threshold and decided value

        Parameters
        ----------
        target_threshold : np.ndarray
            target thresholds to the margin node describing the capability of the design.
            The length of this vector equals the number of samples
        decided_value : np.ndarray
            The decided values that the design needs to achieve
            The length of this vector equals the number of samples
        """

        self.decided_value(decided_value)  # add to list of decided values
        self.target(target_threshold)  # add to list of targets

        if self.direction == 'must_exceed':
            e = target_threshold - decided_value
        elif self.direction == 'must_not_exceed':
            e = decided_value - target_threshold
        else:
            raise Exception(
                'Wrong margin type (%s) specified. Possible values are "must_Exceed", "must_not_exceed".' % (
                    str(self.direction)))

        self.excess(e)

    def __deepcopy__(self, memo): # memo is a dict of id's to copies
        """
        creates a deep independent copy of the class instance self.
        https://stackoverflow.com/a/15774013

        Parameters
        ----------
        memo : Dict
            memoization dictionary of id(original) (or identity numbers) to copies

        Returns
        -------
        Decision
            copy of MarginNode instance
        """
        id_self = id(self) # memoization avoids unnecessary recursion
        _copy = memo.get(id_self)
        if _copy is None:

            _copy = type(self)(self.key, self.cutoff, self.buffer_limit, self.direction)
            _copy.key = self.key
            _copy.target = deepcopy(self.target,memo)
            _copy.decided_value = deepcopy(self.decided_value,memo)
            _copy.excess = deepcopy(self.excess,memo)
            memo[id_self] = _copy 
        return _copy


class Behaviour():
    def __init__(self, n_i: int, n_p: int, n_dv: int, n_tt: int, key: str = ''):
        """
        This class stores the method for calculating its outputs
            - Intermediate parameters
            - Performance parameters
            - Decided values
            - target thresholds

        Parameters
        ----------
        n_i : int
            number of intermediate parameters
        n_p : int
            number of performance parameters
        n_dv : int
            number of decided values
        n_tt : int
            number of target threshold
        key : str, optional
            string to tag instance with, default = ''
        """
        self.key = key
        self.n_i = n_i
        self.n_p = n_p
        self.n_dv = n_dv
        self.n_tt = n_tt
        self._intermediate = np.empty(self.n_i)
        self._performance = np.empty(self.n_p)
        self._decided_value = np.empty(self.n_dv)
        self._threshold = np.empty(self.n_tt)
        self._inverted = np.empty(1)
        self.surrogate_available = False
        self.xt = None
        self.yt = None
        self.sm = None
        self.variable_dict = {}
        self.xtypes = []
        self.xlimits = []
        self.lb_outputs = None
        self.ub_outputs = None
        self.surrogate_inv_available = False
        self.xt_inv = None
        self.yt_inv = None
        self.sm_inv = None
        self.xtypes_inv = []
        self.xlimits_inv = []
        self.lb_outputs_inv = None
        self.ub_outputs_inv = None

    @property
    def intermediate(self) -> Union[float,List[float]]:
        """
        intermediate parameters getter

        Returns
        -------
        Union[float,List[float]]
            vector of intermediate parameters
        """
        if self.n_i == 1:
            return self._intermediate[0]
        else:
            return list(self._intermediate)

    @intermediate.setter
    def intermediate(self, i: Union[float,List[float],np.ndarray]):
        """
        Sets the intermediate parameters as a numpy array

        Parameters
        ----------
        i : Union[float,List[float],np.ndarray]
            value to set intermediate parameters
        """
        if isinstance(i,(int,float,np.floating)):
            assert self.n_i == 1
            self._intermediate = np.array([i,])
        else:
            assert len(i) == self.n_i, 'expected %i parameters, got %i parameters' %(self.n_i,len(i))
            self._intermediate = np.array(i)

    @property
    def performance(self) -> Union[float,List[float]]:
        """
        performance parameters getter

        Returns
        -------
        np.ndarray
            vector of performance parameters
        """
        if self.n_p == 1:
            return self._performance[0]
        else:
            return list(self._performance)

    @performance.setter
    def performance(self, i: Union[float,List[float],np.ndarray]):
        """
        Sets the performance parameters as a numpy array

        Parameters
        ----------
        i : Union[float,List[float],np.ndarray]
            value to set performance parameters
        """
        if isinstance(i,(int,float,np.floating)):
            assert self.n_p == 1
            self._performance = np.array([i,])
        else:
            assert len(i) == self.n_p, 'expected %i parameters, got %i parameters' %(self.n_p,len(i))
            self._performance = np.array(i)

    @property
    def decided_value(self) -> Union[float,List[float]]:
        """
        decided_value parameters getter

        Returns
        -------
        np.ndarray
            vector of decided_value parameters
        """
        if self.n_dv == 1:
            return self._decided_value[0]
        else:
            return list(self._decided_value)

    @decided_value.setter
    def decided_value(self, i: Union[float,List[float],np.ndarray]):
        """
        Sets the decided_value parameters as a numpy array

        Parameters
        ----------
        i : Union[float,List[float],np.ndarray]
            value to set decided_value parameters
        """
        if isinstance(i,(int,float,np.floating)):
            assert self.n_dv == 1
            self._decided_value = np.array([i,])
        else:
            assert len(i) == self.n_dv, 'expected %i parameters, got %i parameters' %(self.n_dv,len(i))
            self._decided_value = np.array(i)

    @property
    def threshold(self) -> Union[float,List[float]]:
        """
        threshold parameters getter

        Returns
        -------
        np.ndarray
            vector of threshold parameters
        """
        if self.n_tt == 1:
            return self._threshold[0]
        else:
            return list(self._threshold)

    @threshold.setter
    def threshold(self, i: Union[float,List[float],np.ndarray]):
        """
        Sets the threshold parameters as a numpy array

        Parameters
        ----------
        i : Union[float,List[float],np.ndarray]
            value to set threshold parameters
        """
        if isinstance(i,(int,float,np.floating)):
            assert self.n_tt == 1
            self._threshold = np.array([i,])
        else:
            assert len(i) == self.n_tt, 'expected %i parameters, got %i parameters' %(self.n_tt,len(i))
            self._threshold = np.array(i)

    @property
    def inverted(self) -> Union[float,List[float]]:
        """
        inverted parameters getter

        Returns
        -------
        np.ndarray
            vector of inverted parameters
        """
        return self._inverted[0]

    @inverted.setter
    def inverted(self, i: Union[float,List[float],np.ndarray]):
        """
        Sets the inverted parameters as a numpy array

        Parameters
        ----------
        i : Union[float,List[float],np.ndarray]
            value to set inverted parameters
        """
        if isinstance(i,(int,float,np.floating)):
            self._inverted = np.array([i,])
        else:
            assert len(i) == 1, 'expected %i parameters, got %i parameters' %(1,len(i))
            self._inverted = np.array(i)

    def train_surrogate(self,variable_dict: Dict[str, Dict[str,Union[Tuple[int,int], Tuple[float,float], List[str], str]] ],
                        n_samples: int, bandwidth: List[float] = [0.01], num_threads: int = 1, sm_type: float = 'KRG', 
                        random_state: Union[int,np.random.RandomState] = None, *args, **kwargs):
        """
        trains a Kriging surrogate model of the behaviour model to make computations less expensive
        surrogate model returns an array of shape 
        [n_samples , (n_intermediate + n_performance + n_decided_value + n_target_threshold) ]
        the inputs are 
        [n_samples , other arguments ]

        Parameters
        ----------
        variable_dict : Dict[str, Dict[str,Union[Tuple[int,int], Tuple[float,float], List[str], str]] ]
            A dictionary with the input variables as keys and their corresponding type ('INT','FLOAT','ENUM')
            and their limits a tuple pair for the upper and lower limits or a list of ints or strings for ENUM variables
        n_samples : int
            number of samples drawn from the behaviour model for training
        bandwidth : List[float], optional
            bandwidth of the correlation function used by Kriging, by default [0.01]
        sm_type : str, optional
            type of surrogate model to train, possible values ['KRG','LS'], by default 'KRG'
        num_threads : int, optional
            number of threads to parallelize sampling the training data by, by default 1
        random_state : Union[int,np.random.RandomState], optional
            Random seed or random generator instance used to generate training samples, 
            If None is provided a random seed is used, by default None
        """
        for key,value in variable_dict.items():
            assert value['type'] in ['FLOAT','INT','ENUM','fixed'], 'Unexpected type (%s) for variable (%s) provided \
                , expecting "FLOAT", "INT", or "ENUM"' %(value['type'], key)

            if value['type'] in ['INT', 'FLOAT']:
                assert type(value['limits']) in [list,tuple], 'Unexpected limits (%s) for variable (%s) provided \
                    , expecting a list of ints or floats' %(str(value['limits']), key)
                if value['type'] == 'INT':
                    assert all([isinstance(l,int) for l in variable_dict[key]['limits']]), \
                        'only int limits are supported, variable %s contains strings in its limits' %(key)
                if value['type'] == 'FLOAT':
                    assert all([isinstance(l,(int,float,np.floating)) for l in variable_dict[key]['limits']]), \
                        'only float limits are supported, variable %s contains strings in its limits' %(key)
                assert len(value['limits']) == 2, 'limits for variable (%s) must be a \
                    pair of ints or floats' %key

            elif value['type'] in ['ENUM']:
                assert type(value['limits']) == list, 'Unexpected limits (%s) for variable (%s) provided \
                , expecting a list of strings or ints' %(str(value['limits']), key)
                assert len(value['limits']) > 0, 'limits for variable (%s) must be a \
                list with at least one element' %key

            elif value['type'] in ['fixed']:
                assert type(value['limits']) in [int,str,float], 'Unexpected value (%s) for variable (%s) provided \
                    , expecting a string, int, or float' %(str(value['limits']), key)

        self.variable_dict = variable_dict

        xtypes = []
        xlimits = []
        for key,value in self.variable_dict.items():
            if value['type'] != 'fixed':
                if value['type'] == 'FLOAT':
                    xtypes += [FLOAT, ]
                elif value['type'] == 'INT':
                    xtypes += [ORD, ]
                elif value['type'] == 'ENUM':
                    xtypes += [(ENUM, len(value['limits'])), ]

                xlimits += [value['limits']]
                
            else:
                args += (value['limits'],)

        assert len(xlimits) > 0, 'at least one variable should not be fixed'

        self.xtypes = xtypes
        self.xlimits = xlimits
        sampling = MixedIntegerSamplingMethod(xtypes=self.xtypes, xlimits=self.xlimits,
                                                sampling_method_class=LHS, criterion="ese",
                                                random_state=random_state)

        input_samples = sampling(n_samples)

        # Parallel computation if num_threads > 1
        behaviour_objs = []
        for pid in range(num_threads):
            behaviour_objs += [deepcopy(self)]

        vargs_iterator = [s for s in input_samples]
        vkwargs_iterator = [{},] * len(input_samples)
        fargs = args
        fkwargs = kwargs
        fkwargs['behaviours'] = behaviour_objs
        fkwargs['variable_dict'] = self.variable_dict

        results = parallel_sampling(_sample_behaviour,vargs_iterator,vkwargs_iterator,fargs,fkwargs,num_threads=num_threads)

        self.xt = np.empty((0,len(self.xtypes)))
        self.yt = np.empty((0,self.n_i+self.n_p+self.n_dv+self.n_tt))
        # Retrieve sampling results
        for i,(sample,result) in enumerate(zip(input_samples,results)):

            # intermediate, performance, decided_value, threshold = result
            
            isnan = False
            for r in result:
                if isinstance(r,(list,np.ndarray)):
                    if np.isnan(r).any():
                        isnan = True
                else:
                    if np.isnan(r):
                        isnan = True

            if isnan:
                print('NaN encountered, skipping point %i' %i)
                continue

            # concatenate input specifications
            sample = sample.reshape(1,self.xt.shape[1])
            result = np.concatenate((result)).reshape(1,self.yt.shape[1])

            self.xt = np.vstack((self.xt, sample))
            self.yt = np.vstack((self.yt, result))

        # Get lower and upper bounds of output values
        self.lb_outputs = np.min(self.yt, axis=0)
        self.ub_outputs = np.max(self.yt, axis=0)

        if sm_type == 'LS': # LS
            surrogate = LS(print_prediction=False)
            xt = self.xt
            yt = self.yt
        elif sm_type == 'KRG': # Kriging surrogate
            surrogate = KRG(theta0=bandwidth, print_prediction=False)
            xt = self.xt[:self.xt.shape[1]*100,:] # truncate training data to avoid bad Kriging Matrix
            yt = self.yt[:self.xt.shape[1]*100,:] # truncate training data to avoid bad Kriging Matrix
        else:
            Exception('Wrong model %s chosen, must provide either "KRG" or "LS"' %sm_type)

        self.sm = MixedIntegerSurrogateModel(
            xtypes=self.xtypes, xlimits=self.xlimits, surrogate=surrogate
        )
        self.sm.set_training_values(xt, yt)
        self.sm.train()
        self.surrogate_available = True

    def train_inverse(self,key: str = None, bandwidth: List[float] = [0.01], sm_type: float = 'KRG'):
        """
        trains a surrogate model to invert decided values back to a chosen input variable and the intermediate parameters
        surrogate model returns an array of shape 
        [n_samples , (optional variables + n_intermediate) ]
        the inputs are 
        [n_samples , (decided value + other arguments) ]

        Parameters
        ----------
        key : str
            unique key string for the variable to be inverted
        bandwidth : List[float], optional
            bandwidth of the correlation function used by Kriging, by default [0.01]
        sm_type : str, optional
            type of surrogate model to train, possible values ['KRG','LS'], by default 'KRG'
        """

        assert self.surrogate_available, 'surrogate must be trained first'
        if key is not None:
            assert all([isinstance(l,(float,int)) for l in self.variable_dict[key]['limits']]), \
                'only float or int limits are supported, variable %s contains strings in its limits' %(key)

        keys = list(self.variable_dict.keys())
        i_index = self.n_i
        p_index = self.n_i+self.n_p
        dv_index = self.n_i+self.n_p+self.n_dv

        self.xt_inv = np.empty((self.xt.shape[0],0))
        self.yt_inv = np.empty((self.xt.shape[0],0))
        self.xtypes_inv = []
        self.xlimits_inv = []

        if key is not None:
            xt_inv_key = np.delete(self.xt, keys.index(key), axis=1) # delete the column of the variable
            yt_inv_key = self.xt[:,keys.index(key)].reshape(self.xt.shape[0],-1)
            self.xt_inv = np.hstack((self.xt_inv, xt_inv_key))
            self.yt_inv = np.hstack((self.yt_inv, yt_inv_key))
            self.xtypes_inv = [x for i,x in enumerate(self.xtypes) if i!=keys.index(key)] 
            self.xlimits_inv = [x for i,x in enumerate(self.xlimits) if i!=keys.index(key)]

        self.xt_inv = np.hstack((self.yt[:,p_index:dv_index], self.xt_inv)) # add decided values columns in front
        self.yt_inv = np.hstack((self.yt_inv, self.yt[:,0:i_index])) # add intermediate columns

        self.xtypes_inv = [FLOAT,] + self.xtypes_inv
        self.xlimits_inv = [[min(self.yt[:,p_index:dv_index])[0], max(self.yt[:,p_index:dv_index])[0]],] + self.xlimits_inv # add decided values limits

        # Get lower and upper bounds of output values
        self.lb_outputs_inv = np.min(self.yt_inv, axis=0)
        self.ub_outputs_inv = np.max(self.yt_inv, axis=0)

        if sm_type == 'LS': # LS
            surrogate = LS(print_prediction=False)
            xt_inv = self.xt_inv
            yt_inv = self.yt_inv
        elif sm_type == 'KRG': # Kriging surrogate
            surrogate = KRG(theta0=bandwidth, print_prediction=False)
            xt_inv = self.xt_inv[:self.xt_inv.shape[1]*100,:] # truncate training data to avoid bad Kriging Matrix
            yt_inv = self.yt_inv[:self.xt_inv.shape[1]*100,:] # truncate training data to avoid bad Kriging Matrix
        elif sm_type == 'QP': # Kriging surrogate
            surrogate = QP(print_prediction=False)
            xt_inv = self.xt_inv
            yt_inv = self.yt_inv
        else:
            Exception('Wrong model %s chosen, must provide either "KRG" or "LS"' %sm_type)

        if all([x == FLOAT for x in self.xtypes_inv]):
            self.sm_inv = surrogate
        else:
            self.sm_inv = MixedIntegerSurrogateModel(
                xtypes=self.xtypes_inv, xlimits=self.xlimits_inv, surrogate=surrogate
            )
        self.sm_inv.set_training_values(xt_inv, yt_inv)
        self.sm_inv.train()
        self.surrogate_inv_available = True 

    def reset(self):
        """
        Resets the stored variables
        """
        self._intermediate = np.empty(self.n_i)
        self._performance = np.empty(self.n_p)
        self._decided_value = np.empty(self.n_dv)
        self._threshold = np.empty(self.n_tt)
        self._inverted = np.empty(0)

    def save(self,filename='man'):
        """
        saves the Behaviour's surrogate model and samples
        Performance surrogate and decision univere
        Saves any stored matrices

        Parameters
        ----------
        filename : str, optional
           basefile path, by default 'man'
        """
        with open(filename+'_behaviour_%s.pkl' % self.key,'wb') as f:
            pickle.dump(self.surrogate_available,f)
            pickle.dump(self.xt,f)
            pickle.dump(self.yt,f)
            pickle.dump(self.sm,f)
            pickle.dump(self.variable_dict,f)
            pickle.dump(self.xtypes,f)
            pickle.dump(self.xlimits,f)
            pickle.dump(self.lb_outputs,f)
            pickle.dump(self.ub_outputs,f)
            pickle.dump(self.surrogate_inv_available,f)
            pickle.dump(self.xt_inv,f)
            pickle.dump(self.yt_inv,f)
            pickle.dump(self.sm_inv,f)
            pickle.dump(self.xtypes_inv,f)
            pickle.dump(self.xlimits_inv,f)
            pickle.dump(self.lb_outputs_inv,f)
            pickle.dump(self.ub_outputs_inv,f)
            pickle.dump(self.n_i,f)
            pickle.dump(self.n_p,f)
            pickle.dump(self.n_dv,f)
            pickle.dump(self.n_tt,f)

    def load(self,filename='man'):
        """
        loads the Behaviour's surrogate model and samples
        Performance surrogate and decision univere
        Saves any stored matrices

        Parameters
        ----------
        filename : str, optional
           basefile path, by default 'man'
        """
        with open(filename+'_behaviour_%s.pkl' % self.key,'rb') as f:
            self.surrogate_available = pickle.load(f)
            self.xt = pickle.load(f)
            self.yt = pickle.load(f)
            self.sm = pickle.load(f)
            self.variable_dict = pickle.load(f)
            self.xtypes = pickle.load(f)
            self.xlimits = pickle.load(f)
            self.lb_outputs = pickle.load(f)
            self.ub_outputs = pickle.load(f)
            self.surrogate_inv_available = pickle.load(f)
            self.xt_inv = pickle.load(f)
            self.yt_inv = pickle.load(f)
            self.sm_inv = pickle.load(f)
            self.xtypes_inv = pickle.load(f)
            self.xlimits_inv = pickle.load(f)
            self.lb_outputs_inv = pickle.load(f)
            self.ub_outputs_inv = pickle.load(f)
            self.n_i = pickle.load(f)
            self.n_p = pickle.load(f)
            self.n_dv = pickle.load(f)
            self.n_tt = pickle.load(f)

    def __call__(self, *args):
        """
        The function that will be used to calculate the outputs of the behaviour model
            - Can be a deterministic model
            - Can be a stochastic model (by calling a defined mvm.Distribution instance)
        This method must be redefined by the user for every instance
        If planning to use a surrogate, the user should extend this method using super()

        Example
        -------
        >>> # [in the plugin file]
        >>> from mvm import Behaviour
        >>> class MyBehaviour(Behaviour):
        >>>     def __call__(self,r,d):
        >>>         # some specific model-dependent behaviour
        >>>         self.intermediate = d
        >>>         self.performance = r*2+1 / d
        >>>         self.decided_value = r**2
        >>>         self.threshold = r/d

        Example
        -------
        >>> # [in the plugin file]
        >>> from mvm import Behaviour
        >>> class MyBehaviour(Behaviour):
        >>>     def __call__(self,r,d,y):
        >>>         args = [r,d]
        >>>         if self.surrogate_available:
        >>>             # the function will terminate here
        >>>             return MyBehaviour.__call__(self,*args)
        >>>         # some specific model-dependent behaviour
        >>>         self.intermediate = d * y
        >>>         self.performance = r*2+1 / d * y
        >>>         self.decided_value = r**2 * y
        >>>         self.threshold = r/d * y
        """
        # default code for the default behaviour
        if self.surrogate_available:
            outputs = self._bounded_pred(np.array(args[:len(self.xtypes_inv)]).reshape(1,-1))[0]
            self._intermediate = outputs[0:self.n_i]
            self._performance = outputs[self.n_i:self.n_i+self.n_p]
            self._decided_value = outputs[self.n_i+self.n_p:self.n_i+self.n_p+self.n_dv]
            self._threshold = outputs[self.n_i+self.n_p+self.n_dv:self.n_i+self.n_p+self.n_dv+self.n_tt]

    def inv_call(self,decided_value: float,*args):
        """
        The function that will be used to calculate the decided values based on a selected variable
        This method must be redefined by the user for every instance, otherwise 
        a surrogate is used if the user has used the ``train_surrogate`` method earlier

        Parameters
        ----------
        decided_value : float
            the decided value to invert

        Example
        -------
        >>> # [in the plugin file]
        >>> from mvm import Behaviour
        >>> class MyBehaviour(Behaviour):
        >>>     def inv_call(self,decided_value,r):
        >>>         # some specific model-dependent behaviour
        >>>         self.intermediate = decided_value
        >>>         self.inverted = r**2
        """

        i_index = self.n_i
        # default code for the default behaviour
        if self.surrogate_inv_available:
            inputs = [decided_value,]+list(args[:len(self.xtypes_inv)-1])
            inputs = np.array(inputs,dtype=object)
            outputs = self._bounded_pred(inputs.reshape(1,-1),'inv')[0]
            self._intermediate = outputs[0:i_index]
            self._inverted = outputs[i_index:]

    def _bounded_pred(self,inputs: np.ndarray, type: str = 'base') -> np.ndarray:
        """
        Retrieve performance estimates from the surrogate

        Parameters
        ----------
        inputs : np.ndarray
            inputs at which to make predictions must be of size
            n_samples x n_inputs
        inputs : np.ndarray, optional
            can 'base' or 'inv', if 'inv used lb_outputs_inv and ub_outputs_inv, by defaul is 'base'

        Returns
        -------
        np.ndarray
            array of estimated performance parameters
        """
        assert type in ['base','inv'], 'type must be either "base" or "inv", got %d' %type

        if type == 'base':
            sm = self.sm
            lb = self.lb_outputs
            ub = self.ub_outputs
        elif type == 'inv':
            sm = self.sm_inv
            lb = self.lb_outputs_inv
            ub = self.ub_outputs_inv

        estimate = sm.predict_values(inputs)
        estimate = np.clip(estimate,lb,ub)
        return estimate

    def __copy__(self):
        """
        returns a shallow copy of Behaviour instance
        https://stackoverflow.com/a/15774013b b b

        Returns
        -------
        Behaviour
            shallow copy of Behaviour instance
        """
        id_self = id(self) # memoization avoids unnecessary recursion
        _copy = type(self)(self.n_i, self.n_p, self.n_dv, self.n_tt, self.key+'_copy_'+str(id(self)))
        if self.surrogate_available:
            _copy.surrogate_available = True
            _copy.xt = self.xt
            _copy.yt = self.yt
            _copy.sm = self.sm
            _copy.variable_dict = self.variable_dict
            _copy.xtypes = self.xtypes
            _copy.xlimits = self.xlimits
            _copy.lb_outputs = self.lb_outputs
            _copy.ub_outputs = self.ub_outputs
            _copy.surrogate_inv_available = self.surrogate_inv_available
            _copy.xt_inv = self.xt_inv
            _copy.yt_inv = self.yt_inv
            _copy.sm_inv = self.sm_inv
            _copy.xtypes_inv = self.xtypes_inv
            _copy.xlimit_inv = self.xlimits_inv
            _copy.lb_outputs_inv = self.lb_outputs_inv
            _copy.ub_outputs_inv = self.ub_outputs_inv
            _copy.n_i = self.n_i
            _copy.n_p = self.n_p
            _copy.n_dv = self.n_dv
            _copy.n_tt = self.n_tt
        return _copy
    
    def __deepcopy__(self, memo): # memo is a dict of id's to copies
        """
        creates a deep independent copy of the class instance self.
        https://stackoverflow.com/a/15774013

        Parameters
        ----------
        memo : Dict
            memoization dictionary of id(original) (or identity numbers) to copies

        Returns
        -------
        Behaviour
            copy of Behaviour instance
        """
        id_self = id(self) # memoization avoids unnecessary recursion
        _copy = memo.get(id_self)
        if _copy is None:

            _copy = type(self)(self.n_i, self.n_p, self.n_dv, self.n_tt, self.key)
            _copy.key = self.key
            if self.surrogate_available:
                _copy.surrogate_available = True
                _copy.xt = deepcopy(self.xt,memo)
                _copy.yt = deepcopy(self.yt,memo)
                _copy.sm = deepcopy(self.sm,memo)
                _copy.variable_dict = deepcopy(self.variable_dict,memo)
                _copy.xtypes = deepcopy(self.xtypes,memo)
                _copy.xlimits = deepcopy(self.xlimits,memo)
                _copy.lb_outputs = deepcopy(self.lb_outputs)
                _copy.ub_outputs = deepcopy(self.ub_outputs)
                _copy.surrogate_inv_available = self.surrogate_inv_available
                _copy.xt_inv = deepcopy(self.xt_inv,memo)
                _copy.yt_inv = deepcopy(self.yt_inv,memo)
                _copy.sm_inv = deepcopy(self.sm_inv,memo)
                _copy.xtypes_inv = deepcopy(self.xtypes_inv,memo)
                _copy.xlimits_inv = deepcopy(self.xlimits_inv,memo)
                _copy.lb_outputs_inv = deepcopy(self.lb_outputs_inv)
                _copy.ub_outputs_inv = deepcopy(self.ub_outputs_inv)
                _copy.n_i = self.n_i
                _copy.n_p = self.n_p
                _copy.n_dv = self.n_dv
                _copy.n_tt = self.n_tt
            memo[id_self] = _copy 
        return _copy


class Decision:
    def __init__(self, universe: Union[Tuple[int,int],List[Union[int, str]]], variable_type: str, key: str = '',
                 direction: Union[str,List[str]] = 'must_exceed',decided_value_model: Behaviour = None, 
                 n_nodes=1, description: str = ''):
        """
        Provide the correct decided value based on a supplied target threshold

        Parameters
        ----------
        universe : Union[Tuple[int,int],List[Union[int, str]]]
            the possible list of discrete values (for off-the-shelf components)
            if variable type is 'INT' this must be a list of 2 integers [max,min]
            if variable type is 'ENUM' this must be a list of integers of strings with legnth > 1
        variable_type : str
            type of variable, possible values are 'INT', 'ENUM'
        key : str, optional
            string to tag instance with, default = ''
        direction : Union[str,List[str]], optional
            possible values('must_exceed','must_not_exceed'),
            if ``n_nodes`` is > 1 then direction must a list of strings with same legnth,
            by default 'must_exceed'
        decided_value_model : Behaviour, optional
            If supplied is used to convert selected_values to decided values,
            otherwise selected value = decided value, by default None
        n_nodes : int, optional
            number of decided value target threshold pairs, default = 1
        description : str, optional
            description string, by default ''
        """

        assert variable_type in ['INT','ENUM'], 'Unexpected variable_type (%s) provided, expecting "INT","ENUM"' %variable_type

        if variable_type == 'INT':
            assert len(universe) == 2, 'for integer type variables you must supple a [min,max] pair for universe'
            assert universe[0] < universe[1], 'max value must be greater than min value'
            assert all([type(v) in [int,float] for v in universe]), 'only int or float values \
                are accepted for `INT` type variable limits'
        elif variable_type == 'ENUM':
            assert len(universe) > 0, 'Universe must have a length of at least 1, %i given' % len(universe)

        assert type(n_nodes) in [float,int], 'n_nodes (%s) must be a float or an integer' %str(n_nodes)

        if isinstance(n_nodes,float):
            assert n_nodes.is_integer(), 'supplied n_nodes (%f) must be a whole number' %n_nodes
            assert n_nodes > 0, 'n_nodes must be greater than zero'
        elif isinstance(n_nodes,int):
            assert n_nodes > 0, 'n_nodes must be greater than zero'
        else:
            Exception('n_nodes must be a float or an int')

        if decided_value_model is not None:
            assert decided_value_model.n_dv == n_nodes, 'decided value model should return n_nodes decided values, \
                got %i, expected %i decided values' %(decided_value_model.n_dv,n_nodes)

        self.variable_type = variable_type
        self.universe = universe
        self.key = key
        self.decided_value_model = decided_value_model
        self.n_nodes = n_nodes
        self.description = description
        self.direction = direction
        self._decided_value = np.zeros(self.n_nodes)
        self._threshold = np.zeros(self.n_nodes)
        self._selection_value = None

        self.selection_values = ParamFactory.build_param(key=self.key, dims=[], datatype='categorical')

        self.decided_values = None
        self.output_value = None
        self.i_min = None

        self.signs = np.empty(0)
        for d in self._direction:
            if d == 'must_exceed':
                self.signs = np.append(self.signs,1.0)
            else:
                self.signs = np.append(self.signs,-1.0)

    @property
    def universe(self) -> Union[Tuple[int,int],List[str]]:
        """
        universe getter

        Returns
        -------
        Union[Tuple[int,int],List[str]]
            direction string(s)
        """

        if self.variable_type == 'INT':
            return [self._universe[0],self._universe[-1],]
        elif self.variable_type == 'ENUM':
            return self._universe
        else:
            return None

    @universe.setter
    def universe(self, u: Union[Tuple[int,int],List[str]]):
        """
        universe setter

        Parameters
        ----------
        u : Union[Tuple[int,int],List[str]]
            universe range or list
        """

        if self.variable_type == 'INT':
            self._universe = list(range(u[0],u[1]+1))
        elif self.variable_type == 'ENUM':
            self._universe = u

    @property
    def direction(self) -> Union[str, List[str]]:
        """
        direction(s) getter

        Returns
        -------
        Union[str, List[str]]
            direction string(s)
        """

        if self.n_nodes == 1:
            return self._direction[0]
        elif self.n_nodes > 1:
            return self._direction
        else:
            return None

    @direction.setter
    def direction(self, d: Union[str, List[str]]):
        """
        checks direction input is correct and converts it to a list

        Parameters
        ----------
        d : Union[str, List[str]]
            direction string(s)
        """

        if self.n_nodes == 1:
            assert type(d) == str, 'only a single direction string is needed for single decision nodes'
            _direction = [d,]
        else:
            assert type(d) == list, 'a list of direction strings with length n_nodes is needed'
            assert len(d) == self.n_nodes, 'a list of direction strings with length n_nodes is needed'
            _direction = d

        assert all([d in ['must_exceed','must_not_exceed'] for d in _direction]), 'Wrong margin type (%s) specified. \
            Possible values are "must_exceed", "must_not_exceed".' % (str(d))

        self._direction = _direction

    @property
    def decided_value(self) -> Union[float,List[float]]:
        """
        decided_value parameters getter

        Returns
        -------
        np.ndarray
            vector of decided_value parameters
        """
        if self.n_nodes == 1:
            return self._decided_value[0]
        else:
            return list(self._decided_value)

    @decided_value.setter
    def decided_value(self, i: Union[float,List[float],np.ndarray]):
        """
        Sets the decided_value parameters as a numpy array

        Parameters
        ----------
        i : Union[float,List[float],np.ndarray]
            value to set decided_value parameters
        """
        if isinstance(i,(int,float,np.floating,np.integer)):
            assert self.n_nodes == 1
            self._decided_value = np.array([i,])
        else:
            assert len(i) == self.n_nodes, 'expected %i parameters, got %i parameters' %(self.n_nodes,len(i))
            self._decided_value = np.array(i)

    @property
    def selection_value(self) -> Union[float,str,int]:
        """
        selection_value parameters getter

        Returns
        -------
        Union[float,str,int]
            vector of selection_value parameters
        """

        return self._selection_value

    @selection_value.setter
    def selection_value(self, i: Union[float,str,int]):
        """
        Sets the selection_value parameter

        Parameters
        ----------
        i : Union[float,str,int]
            value to set selection_value parameters
        """

        if i is not None:
            assert i in self._universe, 'invalid selection provided, \
                value must be in universe %s or equal to None, got %d' %(','.join(map(str, self._universe)),i)

            if self.variable_type == 'INT':
                assert isinstance(i,(int,np.integer)) or i.is_integer(), \
                    'selection value must be of type int'
                if isinstance(i,(float,np.floating)):
                    self._selection_value = int(i)
                else:
                    self._selection_value = int(i)
            elif self.variable_type == 'ENUM':
                assert isinstance(i,(int,np.integer,float,np.floating,str)), 'selection value must be of type str'
                self._selection_value = i

    @property
    def threshold(self) -> Union[float,List[float]]:
        """
        threshold parameters getter

        Returns
        -------
        np.ndarray
            vector of threshold parameters
        """
        if self.n_nodes == 1:
            return self._threshold[0]
        else:
            return list(self._threshold)

    @threshold.setter
    def threshold(self, i: Union[float,List[float],np.ndarray]):
        """
        Sets the threshold parameters as a numpy array

        Parameters
        ----------
        i : Union[float,List[float],np.ndarray]
            value to set threshold parameters
        """
        if isinstance(i,(int,float,np.integer,np.floating)):
            assert self.n_nodes == 1
            self._threshold = np.array([i,])
        else:
            assert len(i) == self.n_nodes, 'expected %i parameters, got %i parameters' %(self.n_nodes,len(i))
            self._threshold = np.array(i)

    def compute_decided_values(self, num_threads: int = 1, *args, **kwargs) -> np.ndarray:
        """
        Converts selected values to decided values

        Parameters
        ----------
        num_threads : int, optional
            number of threads to parallelize sampling process, by default 1

        Returns
        ----------
        np.ndarray
            Vector of decided values after conversion
        """

        if self.decided_value_model is not None:

            # Parallel computation if num_threads > 1
            behaviours = []
            for pid in range(num_threads):
                behaviours += [deepcopy(self.decided_value_model)]

            vargs_iterator = [[value,] for value in self._universe]
            vkwargs_iterator = [{},] * len(self._universe)
            fargs = args
            fkwargs = kwargs
            fkwargs['behaviours'] = behaviours

            results = parallel_sampling(_sample_behaviour,vargs_iterator,vkwargs_iterator,fargs,fkwargs,num_threads=num_threads)

            decided_values = np.empty((0,self.n_nodes))
            # Retrieve sampling results
            for result in results:
                # [intermediate, performance, decided_value, threshold] = results
                decided_value = result[2]
                assert(len(decided_value) == self.n_nodes)

                decided_values = np.vstack((decided_values, decided_value))

            self.decided_values = decided_values
        else:
            assert all([type(x) != str for x in self._universe]), \
                'Decided value model must be provided to convert a non-int or float universe'
            self.decided_values = np.array(self._universe).reshape(-1,1)

        return self.decided_values

    def allocate_margin(self, target_threshold: Union[int, float, List[int], List[float]], 
                        strategy: str = 'min_excess'):
        """
        Calculate the nearest decided value that yield a positive margin

        Parameters
        ----------
        target_threshold : Union[int, float, List[int], List[float]]
            target threshold from intermediate calculations that feeds the margin node describing the capability
            of the design.
        strategy : str, optional
            Can be of value 'min_exces', 'manual'. If 'manual', the self.decision_vector must be defined earlier, 
            by default 'min_excess'
        """

        assert self.decided_values is not None, 'Decided values have not been computed. \
            Use the `compute_decided_values` method or the `init_decisions` \
            method of `MarginNetwork` class'

        if strategy == 'min_excess':
            # find the selected value based on minimum excess

            # Compute excess vector for each value in the decision universe
            excesses = np.empty((0,self.n_nodes))
            for decided_value in self.decided_values:
                e = (target_threshold - decided_value) * self.signs
                excesses = np.vstack((excesses, e))

            valid_idx = np.where(np.all(excesses >= 0,axis=1))[0]
            if len(valid_idx) > 0:
                # if at least one excess value is positive get the smallest excess value
                id_min, _ = np.unravel_index(np.argmin(excesses[valid_idx]),excesses[valid_idx].shape)
                i_min = valid_idx[id_min]
            else:
                # if all excess are negative get the largest negative excess
                i_min, _ = np.unravel_index(np.argmax(excesses),excesses.shape)

        elif strategy == 'manual':
            i_min = np.where([self.selection_value == v for v in self._universe])[0][0]

        self.selection_value = self._universe[i_min]
        self.selection_values(self._universe[i_min])  # Store selection_value
        if self.n_nodes == 1:
            self.decided_value = self.decided_values[i_min,0]
        else:
            self.decided_value = list(self.decided_values[i_min,:])
        self.i_min = i_min

    def save(self,filename='man',results_only: bool=False):
        """
        saves the Decisions's surrogate model and decided values,
        selected value,  decided value, and i_min

        Parameters
        ----------
        filename : str, optional
           basefile path, by default 'man'
        results_only : bool, optional
           if True only saves results related objects such as decided_values, 
           selection_value, output_value, i_min, and selection_values, by default False
        """
        
        with open(filename+'_decision_%s.pkl' % self.key,'wb') as f:
            pickle.dump(self.decided_values,f)
            pickle.dump(self.selection_value,f)
            pickle.dump(self.output_value,f)
            pickle.dump(self.i_min,f)
            pickle.dump(self.selection_values,f)

        if self.decided_value_model is not None and not results_only:
            self.decided_value_model.save(filename)

    def load(self,filename='man',results_only: bool=False):
        """
        loads the Decisions's surrogate model and decided values,
        selected value,  decided value, and i_min

        Parameters
        ----------
        filename : str, optional
           basefile path, by default 'man'
        results_only : bool, optional
           if True only loads results related objects such as decided_values, 
           selection_value, output_value, i_min, and selection_values, by default False
        """
        
        with open(filename+'_decision_%s.pkl' % self.key,'rb') as f:
            self.decided_values = pickle.load(f)
            self.selection_value = pickle.load(f)
            self.output_value = pickle.load(f)
            self.i_min = pickle.load(f)
            self.selection_values = pickle.load(f)

        if self.decided_value_model is not None and not results_only:
            self.decided_value_model.load(filename)

    def reset(self):
        """
        Resets the stored variables
        """
        self.decided_values = None
        self.selection_value = None
        self.output_value = None
        self.i_min = None
        self._decided_value = np.empty(self.n_nodes)
        self._threshold = np.empty(self.n_nodes)

    def reset_outputs(self, n: int = None):
        """
        Resets the stored observations for selection_values

        Parameters
        -----------
        n : int, optional
            if provided deletes only the last n_samples, where possible, by default None
        """
        self.selection_values.reset(n)

    def __call__(self, target_threshold: Union[int, float, List[int], List[float]], 
                 recalculate=False, allocate: bool = False, strategy: str = 'min_excess', 
                 num_threads: int = 1, output='dv', *args, **kwargs) -> Tuple[Union[int, float], Union[int, str]]:
        """
        Calculate the nearest decided value that yield a positive margin

        Parameters
        ----------
        target_threshold : Union[int, float, List[int], List[float]]
            target threshold from intermediate calculations that feeds the margin node describing the capability
            of the design.
        recalculate : bool, optional
            if True calculates all the decided values on the universe,
            by default True
        allocate : bool, optional
            if True allocate the decided values using supplied strategy,
            by default False
        strategy : str, optional
            Can be of value 'min_exces', 'manual'. If 'manual', the self.decision_vector must be defined earlier, 
            by default 'min_excess'
        num_threads : int, optional
            number of threads to parallelize decision universe computation, be default 1
        output : str, optional
            whether to return the decided value or the target threshold, be default 'dv'   

        Returns
        -------
        Tuple[Union[int, float], Union[int, str]]
            The selected value from the design parameter and the corresponding decided value
        """
        
        assert output in ['dv','tt'], 'argument output must one of "dv" or "tt"'

        self.threshold = target_threshold

        if recalculate:
            self.compute_decided_values(num_threads, *args, **kwargs)

        if allocate:
            self.allocate_margin(target_threshold,strategy)

        if not allocate and not recalculate:
            assert self.selection_value is not None, 'the selected value for this node was not initialized'
            i_min = np.where([self.selection_value == v for v in self._universe])[0][0]

            if self.decided_value_model is not None:
                self.decided_value_model(self.selection_value,*args,**kwargs)
                decided_value = self.decided_value_model.decided_value

                if self.n_nodes > 1:
                    assert len(decided_value) == self.n_nodes, \
                        'decided value of decision (%s) must be a vector of length n_nodes' % self.key

                self.decided_value = decided_value
            else:
                # recalculate everything
                self.compute_decided_values(num_threads, *args, **kwargs)
                if self.n_nodes == 1:
                    self.decided_value = self.decided_values[i_min,0]
                else:
                    self.decided_value = list(self.decided_values[i_min,:])

            self.i_min = i_min

        if output == 'dv':
            self.output_value = self.decided_value
        else:
            self.output_value = self.threshold

        return self.output_value, self.selection_value

    def __copy__(self):
        """
        returns a shallow copy of Decision instance
        https://stackoverflow.com/a/15774013

        Returns
        -------
        Decision
            shallow copy of Decision instance
        """
        id_self = id(self) # memoization avoids unnecessary recursion
        return type(self)(self.universe, self.variable_type, self.key+'_copy_'+str(id(self)),
                          self.direction, self.decided_value_model,
                          self.n_nodes, self.description)
    
    def __deepcopy__(self, memo): # memo is a dict of id's to copies
        """
        creates a deep independent copy of the class instance self.
        https://stackoverflow.com/a/15774013

        Parameters
        ----------
        memo : Dict
            memoization dictionary of id(original) (or identity numbers) to copies

        Returns
        -------
        Decision
            copy of Decision instance
        """
        id_self = id(self) # memoization avoids unnecessary recursion
        _copy = memo.get(id_self)
        if _copy is None:

            _copy = type(self)(copy(self.universe), self.variable_type, self.key, 
                               copy(self.direction), deepcopy(self.decided_value_model,memo),
                               self.n_nodes, self.description)
            _copy.key = self.key
            memo[id_self] = _copy 
        return _copy


class MarginNetwork():
    def __init__(self, design_params: List[DesignParam], input_specs: List[InputSpec],
                 fixed_params: List[FixedParam], behaviours: List[Behaviour], decisions: List[Decision],
                 margin_nodes: List[MarginNode], performances: List[Performance], key: str = ''):
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
        decisions : List[Decision]
            list of Decision instances
        margin_nodes : List[MarginNode]
            list of MarginNode instances
        margin_nodes : List[Performance]
            list of Performance instances
        distributions : List[Distribution]
            list of Distribution instances
        key : str, optional
            string to tag instance with, default = ''
        """
        self.sm_perf = None
        self.fig = None
        self.lb_inputs = None
        self.ub_inputs = None
        self.lb_outputs = None
        self.ub_outputs = None
        self.key = key

        # Inputs
        self.design_params = design_params
        self.input_specs = input_specs
        self.fixed_params = fixed_params
        self.decisions = decisions
        self.behaviours = behaviours

        # Outputs
        self.margin_nodes = margin_nodes
        self.performances = performances
        self.deterioration_vector = ParamFactory.build_param(key=self.key, dims=[len(input_specs)])
        self.impact_matrix = ParamFactory.build_param(key=self.key, dims=[len(margin_nodes), len(performances), ])
        self.absorption_matrix = ParamFactory.build_param(key=self.key, dims=[len(margin_nodes), len(input_specs), ])
        self.utilization_matrix = ParamFactory.build_param(key=self.key, dims=[len(margin_nodes), len(input_specs), ])

        # Intermediate attributes
        self.xt = np.empty((0, len(self.margin_nodes) + len(self.input_specs)))  # excess + input secs
        self.yt = np.empty((0, len(self.performances)))  # Performance parameters
        self.spec_limit = np.empty(0)
        self.threshold_limit = np.empty((len(self.margin_nodes), 0))
        self.initial_decision = self.decision_vector
        self.n_forwards = 0

        # Design parameter space
        universe_d = []
        variable_type_d = []

        # Get upper and lower bound for continuous variables
        for design_param in self.design_params:
            assert min(design_param.universe) < max(design_param.universe), \
                'max of universe of design parameter %s must be greater than min' % design_param.key

            universe_d += [design_param.universe]
            variable_type_d += [design_param.variable_type]

        # Input specification space
        universe_s = []
        variable_type_s = []

        for input_spec in self.input_specs:
            assert min(input_spec.universe) < max(input_spec.universe), \
                'max of universe of input spec %s must be greater than min' % input_spec.key

            universe_s += [input_spec.universe]
            variable_type_s += [input_spec.variable_type]

        # Decision space
        universe_decision = []
        variable_type_decision = []

        # Get upper and lower bound for continuous variables
        for decision in self.decisions:
            universe_decision += [decision.universe]
            variable_type_decision += [decision.variable_type]

        self.universe_d = universe_d
        self.variable_type_d = variable_type_d

        self.universe_s = universe_s
        self.variable_type_s = variable_type_s

        self.universe_decision = universe_decision
        self.variable_type_decision = variable_type_decision

        self.universe = self.universe_d + self.universe_s + self.universe_decision
        self.variable_type = self.variable_type_d + self.variable_type_s + self.variable_type_decision

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
            vector = np.append(vector, item.value)
        return vector

    @design_vector.setter
    def design_vector(self, d: np.ndarray):
        """
        Adjusts all the design parameters according to the vector d provided

        Parameters
        ----------
        d : ndarray
            value to set the input specs of the MAN to
        """
        assert d.ndim == 1
        assert len(d) == len(self.design_params)

        for value, item in zip(d, self.design_params):
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
            vector = np.append(vector, item.value)
        return vector

    @spec_vector.setter
    def spec_vector(self, d: np.ndarray):
        """
        Adjusts all the input specs according to the  vector d provided

        Parameters
        ----------
        d : ndarray
            value to set the input specs of the MAN to
        """
        assert d.ndim == 1
        assert len(d) == len(self.input_specs)

        for value, item in zip(d, self.input_specs):
            item.value = value

    @property
    def spec_signs(self) -> np.ndarray:
        """
        returns a vector of input specifications directions

        Returns
        -------
        np.ndarray
            vector of input specification change directions
        """
        vector = np.empty(0)
        for item in self.input_specs:
            vector = np.append(vector, np.sign(item.inc))
        return vector

    @property
    def decision_vector(self) -> List[Union[str, int]]:
        """
        returns a vector of decisions

        Returns
        -------
        List[Union[str, int]]
            List of decisions
        """
        vector = []
        for item in self.decisions:
            vector += [item.selection_value]
        return vector

    @decision_vector.setter
    def decision_vector(self, d: List[Union[str, int]]):
        """
        Adjusts all the decisions according to the vector d provided

        Parameters
        ----------
        d : List[Union[str, int]]
            list of values to set the decisions of the MAN to
        """
        assert len(d) == len(self.decisions)

        for value, item in zip(d, self.decisions):
            item.selection_value = value

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
            vector = np.append(vector, item.original)
        return vector

    @property
    def nominal_design_vector(self) -> np.ndarray:
        """
        returns a vector of nominal design parameter

        Returns
        -------
        np.ndarray
            vector of nominal design parameters
        """
        vector = np.empty(0)
        for item in self.design_params:
            vector = np.append(vector, item.original)
        return vector

    @nominal_design_vector.setter
    def nominal_design_vector(self, d: np.ndarray):
        """
        Adjusts all the nominal design parameters according to the vector d provided
        These values are not affected by the ``reset`` method
        Parameters
        ----------
        d : ndarray
            value to set the nominal design parameters of the MAN to
        """
        assert d.ndim == 1
        assert len(d) == len(self.design_params)

        for value, item in zip(d, self.design_params):
            item.original = value

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
            vector = np.append(vector, item.excess.value)
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
            vector = np.append(vector, item.decided_value.value)
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
            vector = np.append(vector, item.target.value)
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
            vector = np.append(vector, item.value)
        return vector

    @property
    def perf_signs(self) -> np.ndarray:
        """
        returns a vector of disarable performance directions

        Returns
        -------
        np.ndarray
            vector of disarable performance directions
        """

        # Get the sign of the performance parameters
        vector = np.empty(0)
        for performance in self.performances:
            if performance.direction == 'less_is_better':
                sign = 1.0
            elif performance.direction == 'more_is_better':
                sign = -1.0
            else:
                sign = 0.0
                Exception('Performance direction : %s is invalid' % str(performance.direction))
            vector = np.append(vector, sign)
        return vector

    def init_decisions(self, num_threads: int = 1):
        """
        Calculates and stores the decided values of all decision nodes in the MAN

        Parameters
        ----------
        num_threads : int, optional
            number of threads to parallelize sampling process, by default 1
        """

        self.forward(recalculate_decisions=True,num_threads=num_threads,outputs=['tt',]*len(self.decisions))  # do not randomize the man for deterioration
        self.reset(n=1)

    def allocate_margins(self, strategy: Union[str,List[str]] = 'min_excess'):
        """
        allocates MAN margins according to some strategy

        Parameters
        ----------
        strategy : Union[str,List[str]], optional
            Can be of value 'min_excess', 'manual' or a list of strings corresponding to each margin node. 
            If 'manual', the self.decision_vector must be defined earlier.
            If a single string is provided all margin nodes are allocated identically,
            by default 'min_excess'
        """
        assert type(strategy) in [list,str], 'argument strategy must be a string or a list of strings'
        if type(strategy) == list:
            assert len(strategy) == len(self.margin_nodes), \
                'list of strings must have the same number of margin nodes, %i provided' %len(strategy)
            assert all([s in ['min_excess','manual'] for s in strategy]), 'only "min_excess","manual" are allowed'
        elif type(strategy) == str:
            assert strategy in ['min_excess','manual'], 'only "min_excess","manual" are allowed'
            strategy = [strategy,] * len(self.margin_nodes)

        self.forward(allocate_margin=True, strategy=strategy)  # do not randomize the man for deterioration
        self.initial_decision = self.decision_vector
        self.reset(n=1)

    def train_performance_surrogate(self, n_samples: int = 100, sampling_freq: int = 1, bandwidth: List[float] = [1e-2], 
                                    sm_type: str = 'KRG', num_threads: int = 1, ext_samples: Tuple[np.ndarray, np.ndarray] = None,
                                    random_state: Union[int,np.random.RandomState] = None):
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
        bandwidth : List[float], optional
            The kernel bandwidth used in Kriging, by default [1e-2,]
        sm_type : str, optional
            type of surrogate model to train, possible values ['KRG','LS'], by default 'KRG'
        num_threads : int, optional
            number of threads to parallelize sampling process, by default 1
        ext_samples : tuple[np.ndarray,np.ndarray], optional
            if sample data provided externally then use directly to fit the response surface, 
            Tuple must have length 2, ext_samples[0] must have shape (N_samples,len(margin_nodes)),
            ext_samples[1] must have shape (N_samples,len(performances)),
            by default None
        random_state : Union[int,np.random.RandomState], optional
            Random seed or random generator instance used to generate training samples, 
            If None is provided a random seed is used, by default None
        """
        if ext_samples is None:
            # generate training data for response surface using an LHS grid of design parameter and input
            # specification space

            input_samples = self._sample_inputs(n_samples,random_state)

            # Parallel computation if num_threads > 1
            man_objs = []
            for pid in range(num_threads):
                man_objs += [deepcopy(self)]

            kwargs = {
                'sampling_freq': sampling_freq,
            }

            vargs_iterator = [[s,] for s in input_samples]
            vkwargs_iterator = [{},] * len(input_samples)
            fargs = []
            fkwargs = kwargs
            fkwargs['mans'] = man_objs

            results = parallel_sampling(_parallel_sample_man,vargs_iterator,vkwargs_iterator,fargs,fkwargs,num_threads=num_threads)

            # Retrieve sampling results
            for result in results:
                
                node_samples, spec, perf_samples = result

                # concatenate input specifications
                x_i = np.append(node_samples, spec)
                x_i = x_i.reshape(1,self.xt.shape[1])

                self.xt = np.vstack((self.xt, x_i))
                self.yt = np.vstack((self.yt, perf_samples.reshape(1, self.yt.shape[1])))

        else:
            assert type(ext_samples) == tuple
            assert len(ext_samples) == 2
            for value in ext_samples:
                assert type(value) == np.ndarray
            assert ext_samples[0].shape[1] == len(self.margin_nodes) + len(self.input_specs)
            assert ext_samples[1].shape[1] == len(self.performances)
            assert ext_samples[0].shape[0] == ext_samples[1].shape[0]

            self.xt = ext_samples[0]
            self.yt = ext_samples[1]

        # Get lower and upper bounds of excess values
        self.lb_inputs = np.min(self.xt, axis=0)
        self.ub_inputs = np.max(self.xt, axis=0)

        # Get lower and upper bounds of performance values
        self.lb_outputs = np.min(self.yt, axis=0)
        self.ub_outputs = np.max(self.yt, axis=0)

        xt = scaling(self.xt, self.lb_inputs, self.ub_inputs, operation=1)
        # xt = self.xt

        self.reset()
        self.reset_inputs()
        assert len(bandwidth) == 1 or len(bandwidth) == xt.shape[1], \
            'bandwidth list size must be at least %i or 1' % xt.shape[1]

        if sm_type == 'LS':
            # LS
            self.sm_perf = LS(print_prediction=False)
            self.sm_perf.set_training_values(xt, self.yt)
            self.sm_perf.train()
        elif sm_type == 'KRG':
            # Kriging surrogate
            self.sm_perf = KRG(theta0=bandwidth, print_prediction=False)
            xt = xt[:self.xt.shape[1]*100,:] # truncate training data to avoid bad Kriging Matrix
            yt = self.yt[:self.xt.shape[1]*100,:] # truncate training data to avoid bad Kriging Matrix
            self.sm_perf.set_training_values(xt, yt)
            self.sm_perf.train()
        else:
            Exception('Wrong model %s chosen, must provide either "KRG" or "LS"' %sm_type)

        """
        # RMTB
        self.sm_perf = RMTB(
            xlimits=bounds,
            order=4,
            num_ctrl_pts=20,
            energy_weight=1e-15,
            regularization_weight=0.0,
        )
        self.sm_perf.set_training_values(xt, self.yt)
        self.sm_perf.train()

        # Mixture of experts
        print("MOE Experts: ", MOE.AVAILABLE_EXPERTS)

        # MOE1: Find the best surrogate model on the whole domain
        self.sm_perf = MOE(n_clusters=1)
        print("MOE1 enabled experts: ", self.sm_perf.enabled_experts)
        self.sm_perf.set_training_values(xt, self.yt)
        self.sm_perf.train()
        """

    def view_perf(self, e_indices: List[int], p_index: int, label_1: str = None, label_2: str = None,
                  label_p: str = None, n_levels: int = 100, folder: str = '', file: str = None,
                  img_format: str = 'pdf'):
        """
        Shows the estimated performance 

        Parameters
        ----------
        e_indices : list[int]
            index of the margin node values to be viewed on the plot
        p_index : int
            index of the performance parameter to be viewed on the plot
        label_1 : str, optional
            axis label of excess value 1if not provided uses the key of MarginNode, 
            by default None
        label_2 : str, optional
            axis label of excess value 2, if not provided uses the key of MarginNode, 
            by default None
        label_p : str, optional
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

        lb_node, ub_node = np.empty(0), np.empty(0)
        for margin_node in self.margin_nodes:
            lb_node = np.append(lb_node, margin_node.decided_value.value)
            ub_node = np.append(ub_node, margin_node.decided_value.value + 1e-3)
        lb_node[e_indices] = self.lb_inputs[e_indices]
        ub_node[e_indices] = self.ub_inputs[e_indices]

        doe_node = Design(lb_node, ub_node, sampling_vector, 'fullfact')
        values = np.hstack((doe_node.unscale(), np.tile(self.nominal_spec_vector, (doe_node.unscale().shape[0], 1))))
        values = scaling(values, self.lb_inputs, self.ub_inputs, operation=1)

        perf_estimate = self._bounded_perf(values)
        # perf_estimate = self.sm_perf.predict_values(values)

        x = doe_node.unscale()[:, e_indices[0]].reshape((n_levels, n_levels))
        y = doe_node.unscale()[:, e_indices[1]].reshape((n_levels, n_levels))
        z = perf_estimate[:, p_index].reshape((n_levels, n_levels))

        if label_1 is None:
            label_1 = 'E_%s' % self.margin_nodes[e_indices[0]].key
        if label_2 is None:
            label_2 = 'E_%s' % self.margin_nodes[e_indices[1]].key
        if label_p is None:
            label_p = self.performances[p_index].key

        ax.contourf(x, y, z, cmap=plt.cm.jet, )
        # ax.plot(self.xt[:50,e_indices[0]],self.xt[:50,e_indices[1]],'.k', markersize = 10) # plot DOE points for surrogate (first 50 only)
        ax.plot(self.xt[:,e_indices[0]],self.xt[:,e_indices[1]],'.k') # plot DOE points for surrogate

        ax.set_xlabel(label_1)
        ax.set_ylabel(label_2)

        cbar = plt.cm.ScalarMappable(cmap=plt.cm.jet)
        cbar.set_array(z)

        boundaries = np.linspace(np.min(perf_estimate[:, p_index]), np.max(perf_estimate[:, p_index]), 51)
        cbar_h = fig.colorbar(cbar, boundaries=boundaries)
        cbar_h.set_label(label_p, rotation=90, labelpad=3)

        if file is not None:
            # Save figure to image
            check_folder('images/%s' % folder)
            self.fig.savefig('images/%s/%s.%s' % (folder, file, img_format),
                             format=img_format, dpi=200, bbox_inches='tight')

        plt.show()

    def compute_impact(self, use_estimate: bool = False):
        """
        Computes the impact on performance (IoP) matrix of the MVM that has a size
        [n_margins, n_performances] and appends its impact matrix if called
        multiple times without a reset(). 
        Must be called after executing at least one `forward()` pass

        Parameters
        ----------
        use_estimate : bool, optional
            uses the trained surrogate model from `train_performance_surrogate` 
            to estimate performance at the threshold design, by default False
        """

        node_values = self.dv_vector.reshape(1, -1)  # turn it into a 2D matrix
        input_specs = self.spec_vector.reshape(1, -1)  # turn it into a 2D matrix

        # Get performances from behaviour models
        performances = self.perf_vector
        performances = np.tile(performances, (len(self.margin_nodes), 1))

        # performances = [len(margin_nodes), len(performances)]

        # Compute performances at target threshold for each margin node
        input_node = np.tile(node_values, (len(self.margin_nodes), 1))  # Create a square matrix

        # input_node = [len(margin_nodes), len(margin_nodes)]

        np.fill_diagonal(input_node, self.tt_vector, wrap=False)  # works in place

        # input_node = [len(margin_nodes), len(margin_nodes)]

        # Alternate decided values
        if use_estimate:
            # concatenate input specifications
            values = np.hstack((input_node, np.tile(self.nominal_spec_vector, (input_node.shape[0], 1))))
            values = scaling(values, self.lb_inputs, self.ub_inputs, operation=1)
            thresh_perf = self._bounded_perf(values)
        else:
            assert len(self.decisions) == len(self.margin_nodes), \
                'All excess margins must have a decision node associated with them, \
                expected %i decision nodes, got %i decision nodes' %(len(self.margin_nodes),len(self.decisions))
            
            thresh_perf = np.empty(performances.shape)
            for i in range(len(self.decisions)):
                outputs = ['dv',] * len(self.decisions)
                outputs[i] = 'tt'
                self.forward(outputs=outputs)
                thresh_perf[i,:] = self.perf_vector

            self.reset(len(self.decisions))

        # thresh_perf = [len(margin_nodes), len(performances)]

        impact = self.perf_signs * (performances - thresh_perf) / thresh_perf

        # impact = [len(margin_nodes), len(performances)]

        self.impact_matrix(impact)  # Store impact matrix

    def compute_absorption(self, num_threads: int = 1, recalculate: bool = True, method: str = "fixed", max_iter: int = 100, **kwargs):
        """
        Computes the change absorption capability (CAC) matrix of the MVM that has a size
        [n_margins, n_input_specs] and appends its absorption matrix if called
        multiple times without a reset(). 
        Also stores and appends the deterioration vectors

        Parameters
        ----------
        num_threads : int, optional
            number of threads to parallelize sampling process, by default 1
        recalculate : bool, optional
            Does not recompute the specification limits or the initial decision vector
        method : bool, optional
            root finding algorithm to use, possible values ['fixed','secant'], by default 'secant'
        max_iter : int, optional
            maximum number of iterations to use during root finding, by default 100
        """

        # Deterioration computation
        
        assert method in ["fixed","secant"], "method %s is invalid, possible values are 'secant', 'secant'" %(method)
        def evaluate_man(a):
            spec.value = a
            # recalculate all the decisions
            self.forward(recalculate_decisions=False,outputs=['tt',]*len(self.decisions),num_threads=num_threads)  # do not randomize the man for deterioration
            self.n_forwards += 1

        def residual_fixed(a):
            evaluate_man(a)
            return min(self.excess_vector)

        def residual_root(a,n):
            evaluate_man(a)
            return self.excess_vector[n]

        if recalculate:
            # Compute target threshold at the spec limit for each margin node
            self.spec_limit = np.empty(0)
            self.threshold_limit = np.empty((len(self.margin_nodes), 0))

            # get nominal values
            self.forward(recalculate_decisions=False,outputs=['tt',]*len(self.decisions),num_threads=num_threads)  # do not randomize the man for deterioration
            self.n_forwards += 1
            nomina_tt_vector = self.tt_vector

            for spec in self.input_specs:

                spec_value = spec.value

                if method == "fixed":
                    # iterate until limit is reached
                    root_value = _fixed_step_method(residual_fixed,spec_value,spec.inc,delta_max=1e3,max_iter=max_iter)
                    if root_value is None:
                        spec_value = min(spec.universe) if spec.inc < 0 else max(spec.universe)
                    else:
                        spec_value = root_value
                    evaluate_man(spec_value)
                    tt_vector = self.tt_vector

                elif method == "secant":

                    root_found = False
                    for node_i in range(len(self.margin_nodes)):
                        root_value = _secant_method(lambda a : residual_root(a,node_i), 
                                                    a=min(spec.universe), b=max(spec.universe),
                                                    lb=min(spec.universe), ub=max(spec.universe),
                                                    max_iter=max_iter)
                        if root_value is None:
                            continue
                        else:
                            evaluate_man(root_value)
                            relative_excess = np.abs(np.min(self.excess_vector/nomina_tt_vector))
                            if relative_excess <= 1e-6:
                                root_found = True
                                break
                            else:
                                root_found = False
                        
                    if not root_found:
                        # iterate until limit is reached
                        root_value = _fixed_step_method(residual_fixed,spec_value,spec.inc,delta_max=1e3,max_iter=max_iter)
                        if root_value is None:
                            spec_value = min(spec.universe) if spec.inc < 0 else max(spec.universe)
                        else:
                            spec_value = root_value
                    else:
                        spec_value = root_value
                        
                evaluate_man(spec_value)
                tt_vector = self.tt_vector

                # retrieve final state
                self.spec_limit = np.append(self.spec_limit, spec_value)
                threshold_limit_vector = np.reshape(tt_vector, (len(self.margin_nodes), -1))
                self.threshold_limit = np.hstack((self.threshold_limit, threshold_limit_vector))
                
                self.reset_inputs()
                self.reset(self.n_forwards)
                self.n_forwards = 0
                self.decision_vector = self.initial_decision

        deterioration = np.max((self.spec_signs * (self.spec_limit - self.nominal_spec_vector) / np.abs(self.nominal_spec_vector),
                                np.zeros(len(self.input_specs))), axis=0)
        deterioration[deterioration == 0] = np.nan  # replace with nans for division

        # Absorption computation

        nominal_threshold = np.reshape(nomina_tt_vector, (len(self.margin_nodes), -1))
        target_thresholds = np.tile(nominal_threshold, (1, len(self.input_specs)))

        # target_thresholds = [len(margin_nodes), len(input_specs)]

        deterioration_matrix = np.tile(deterioration, (len(self.margin_nodes), 1))

        # deterioration_matrix = [len(margin_nodes), len(input_specs)]

        absorption = np.maximum(abs(self.threshold_limit - target_thresholds) / (target_thresholds * deterioration_matrix),
                                np.zeros_like(self.threshold_limit))
        absorption[np.isnan(absorption)] = np.nan  # replace undefined absorptions with 0
        
        # absorption = [len(margin_nodes), len(input_specs)]

        # utilization computation

        decided_value = np.reshape(self.dv_vector, (len(self.margin_nodes), -1))
        decided_values = np.tile(decided_value, (1, len(self.input_specs)))

        # decided_values = [len(margin_nodes), len(input_specs)]

        with np.errstate(divide='ignore',invalid='ignore'):
            utilization = 1 - ((decided_values - self.threshold_limit) / (decided_values - target_thresholds))

        # utilization = [len(margin_nodes), len(input_specs)]

        # store the results
        self.absorption_matrix(absorption)
        self.deterioration_vector(deterioration)
        self.utilization_matrix(utilization)

    def compute_mvp(self, plot_type: str = 'scatter', show_neutral=False) -> float:
        """
        computes the margin value map after running all Monte Carlo computations

        Parameters
        ----------
        plot_type : str, optional
            possible values are ('scatter','mean','density')
            if 'scatter' is selected, simply plots all the sample absorptions and impacts,
            if 'mean' is selected, plots all the mean of all sample absorptions and impacts,
            if 'density' is selected, plots a Gaussian KDE of the sample absorptions and impacts,
            by default True
        show_neutral : bool, optional
            If true displays the distance from the mean and the neutral 45 deg line

        Returns
        -------
        float
            The aggregate distance from the neutral line
        """

        color = np.random.random((100, 3))

        # Extract x and y
        x = np.mean(self.impact_matrix.values,
                    axis=1).ravel()  # average along performance parameters (assumes equal weighting)
        y = np.mean(self.absorption_matrix.values,
                    axis=1).ravel()  # average along input specs (assumes equal weighting)

        c = np.empty((0, 3))
        for i in range(len(self.margin_nodes)):
            c_node = np.tile(color[i], (self.absorption_matrix.values.shape[2], 1))
            c = np.vstack((c, c_node))

        # Define the borders
        deltax = (np.nanmax(x) - np.nanmin(x)) / 10
        deltay = (np.nanmax(y) - np.nanmin(y)) / 10
        xmin = np.nanmin(x) - deltax
        ymax = np.nanmax(y) + deltay
        xmax = np.nanmax(x) + deltax
        ymin = np.nanmin(y) - deltay

        # create empty figure
        fig, ax = plt.subplots(figsize=(7, 8))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel('Impact on performance')
        ax.set_ylabel('Change absoption capability')

        if plot_type == 'scatter':

            ax.scatter(x, y, s=20, c=c)

        elif plot_type == 'mean':

            c = np.empty((0, 3))
            for i in range(len(self.margin_nodes)):
                c = np.vstack((c, color[i]))

            # Extract x and y
            x = np.nanmean(self.impact_matrix.values,
                           axis=(1, 2)).ravel()  # average along performance parameters (assumes equal weighting)
            y = np.nanmean(self.absorption_matrix.values,
                           axis=(1, 2)).ravel()  # average along input specs (assumes equal weighting)

            ax.scatter(x, y, s=50, c=c)

        elif plot_type == 'density':

            # Create meshgrid
            xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

            # fit a Gaussian KDE
            values = np.vstack([x, y])
            values = values[:, ~np.isnan(values).any(axis=0)]  # drop nans
            kernel = st.gaussian_kde(values)

            # Predict at meshgrid points
            positions = np.vstack([xx.ravel(), yy.ravel()])
            f = np.reshape(kernel(positions).T, xx.shape)

            # plot the KDE contours
            # c2 = ax.contourf( X1, X2, Z, colors=['#1EAA37'], alpha=0.0)
            ax.contourf(xx, yy, f, alpha=0.25, cmap=plt.cm.Blues)
            ax.contour(xx, yy, f, colors='b')

        # distance computation
        dist = 0
        x = np.nanmean(self.impact_matrix.values,
                       axis=(1, 2)).ravel()  # average along performance parameters (assumes equal weighting)
        y = np.nanmean(self.absorption_matrix.values,
                       axis=(1, 2)).ravel()  # average along input specs (assumes equal weighting)
        p1 = np.array([xmin, ymin])
        p2 = np.array([xmax, ymax])

        if show_neutral:
            ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='k', linestyle=(5, (10, 5)))

        d: float = 0.0
        for i in range(len(self.margin_nodes)):
            s = np.array([x[i], y[i]])

            pn, d = nearest(p1, p2, s)
            dist += d

            if show_neutral:
                x_d = [s[0], pn[0]]
                y_d = [s[1], pn[1]]
                ax.plot(x_d, y_d, marker='.', linestyle='--', color=color[i])

        plt.show()

        return d

    @abstractmethod
    def randomize(self, *args, **kwargs):
        """
        The function that will be used to randomize the MAN. 
        Include all random distribution objects here and have 
        them draw samples by calling them. See example below

        Example
        -------
        >>> # [in the plugin file]
        >>> from mvm import MarginNetwork, InputSpec, GaussianFunc
        >>> dist_1 = GaussianFunc(1.0,0.1)
        >>> dist_2 = GaussianFunc(0.5,0.2)
        >>> s1 = InputSpec(1.0 ,'S1', distribution=dist_1)
        >>> s2 = InputSpec(0.5 ,'S2', distribution=dist_2)
        >>> class MyMarginNetwork(MarginNetwork):
        >>>     def randomize(self):
        >>>         # randomization procedure
        >>>         s1.random()
        >>>         s2.random()
        >>>     def forward(self):
        >>>         # some specific model-dependent behaviour
        >>>         x = s1.value + s2.value # These values will be drawn from their respective distributions
        """
        # default code for the default threshold
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        The function that will be used to calculate a forward pass of the MAN
        (design_params,input_specs,fixed_params) -> (excess,performance)
        This method must be redefined by the user for every instance

        Example
        -------
        >>> # [in the plugin file]
        >>> from mvm import MarginNetwork
        >>> class MyMarginNetwork(MarginNetwork):
        >>>     def randomize(self):
        >>>         pass
        >>>     def forward(self,num_threads=1,recalculate_decisions=False,allocate_margin=False):
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
        >>>         e1(b2.decided_value,s1.random())
        """
        # default code for the default threshold
        pass

    def reset(self, n: int = None):
        """
        Resets all elements of the MAN by clearing their internal caches

        Parameters
        -----------
        n : int, optional
            if provided deletes only the last n_samples, where possible, by default None
        """

        for behaviour in self.behaviours:
            behaviour.reset()
        for margin_node in self.margin_nodes:
            margin_node.reset(n)
        for performance in self.performances:
            performance.reset(n)
        if n is None:
            for decision in self.decisions:
                decision.reset()

    def reset_inputs(self, n: int = None):
        """
        Resets Design parameters and input specifications

        Parameters
        -----------
        n : int, optional
            if provided deletes only the last n_samples, where possible, by default None
        """

        for design_param in self.design_params:
            design_param.reset()
        for input_spec in self.input_specs:
            input_spec.reset(n)

    def reset_outputs(self, n: int = None):
        """
        Resets Impact, Absorption, Deterioration, and Utilization matrices

        Parameters
        -----------
        n : int, optional
            if provided deletes only the last n_samples, by default None
        """
        
        self.utilization_matrix.reset(n)
        self.deterioration_vector.reset(n)
        self.impact_matrix.reset(n)
        self.absorption_matrix.reset(n)
        for decision in self.decisions:
            decision.reset_outputs(n)

    def save(self,filename: str='man',folder: str='/',results_only: bool=False):
        """
        saves the initial state of the MarginNetwork:
        Performance surrogate and decision univere
        Saves any stored matrices

        Parameters
        ----------
        filename : str, optional
           prefix before saved files, by default 'man'
        folder : str, optional
           basefile path, by default is current directory
        results_only : bool, optional
           if True only saves results related objects such as margin nodes, 
           input specs, performances, and impact and absorption matrices, by default False
        """
        check_folder(folder)
        path = os.path.join(folder,filename)
        
        if not results_only:
            with open(path+'_init'+'.pkl','wb') as f:
                pickle.dump(self.lb_inputs,f)
                pickle.dump(self.ub_inputs,f)
                pickle.dump(self.lb_outputs,f)
                pickle.dump(self.ub_outputs,f)
                pickle.dump(self.xt,f)
                pickle.dump(self.yt,f)
                pickle.dump(self.sm_perf,f)
                pickle.dump(self.initial_decision,f)
                pickle.dump(self.decision_vector,f)

        with open(path+'_samples'+'.pkl','wb') as f:
            pickle.dump(self.deterioration_vector,f)
            pickle.dump(self.impact_matrix,f)
            pickle.dump(self.absorption_matrix,f)
            pickle.dump(self.utilization_matrix,f)

        with open(path+'_specs'+'.pkl','wb') as f:
            for input_spec in self.input_specs:
                pickle.dump(input_spec,f)

        for decision in self.decisions:
            decision.save(filename=path,results_only=results_only)

        for margin_node in self.margin_nodes:
            margin_node.save(filename=path)

        with open(path+'_performances'+'.pkl','wb') as f:
            for performance in self.performances:
                pickle.dump(performance,f)

            margin_node.save(filename=path)

        if not results_only:
            for behaviour in self.behaviours:
                behaviour.save(filename=path)

    def load(self,filename: str='man',folder: str='/',results_only: bool=False):
        """
        loads the initial state of the MarginNetwork:
        Performance surrogate and decision univere
        loads any stored matrices

        Parameters
        ----------
        filename : str, optional
           prefix before loaded files, by default 'man'
        folder : str, optional
           basefile path, by default is current directory
        results_only : bool, optional
           if True only loads results related objects such as margin nodes, 
           input specs, performances, and impact and absorption matrices, by default False
        """
        path = os.path.join(folder,filename)

        if not results_only:
            with open(path+'_init'+'.pkl','rb') as f:
                self.lb_inputs = pickle.load(f)
                self.ub_inputs = pickle.load(f)
                self.lb_outputs = pickle.load(f)
                self.ub_outputs = pickle.load(f)
                self.xt = pickle.load(f)
                self.yt = pickle.load(f)
                self.sm_perf = pickle.load(f)
                self.initial_decision = pickle.load(f)
                self.decision_vector = pickle.load(f)

        with open(path+'_samples'+'.pkl','rb') as f:
            self.deterioration_vector = pickle.load(f)
            self.impact_matrix = pickle.load(f)
            self.absorption_matrix = pickle.load(f)
            self.utilization_matrix = pickle.load(f)

        with open(path+'_specs'+'.pkl','rb') as f:
            for i in range(len(self.input_specs)):
                self.input_specs[i] = pickle.load(f)

        for decision in self.decisions:
            decision.load(filename=path,results_only=results_only)

        for margin_node in self.margin_nodes:
            margin_node.load(filename=path)

        with open(path+'_performances'+'.pkl','rb') as f:
            for p in range(len(self.performances)):
                self.performances[p] = pickle.load(f)

        if not results_only:
            for behaviour in self.behaviours:
                behaviour.load(filename=path)

    def _sample_inputs(self,n_samples: int, random_state: Union[int,np.random.RandomState] = None) -> np.ndarray:
        """
        samples the design space of the MAN

        Parameters
        ----------
        n_samples : int
            number of samples

        Returns
        -------
        np.ndarray
            An array of samples with shape (n_samples,n_variables), 
            where n_variables is the total number of design variables
        random_state : Union[int,np.random.RandomState], optional
            Random seed or random generator instance used to generate training samples, 
            If None is provided a random seed is used, by default None
        """
        xtypes = []
        xlimits = []
        for xtype, limits in zip(self.variable_type, self.universe):
            if xtype == 'FLOAT':
                xtypes += [FLOAT, ]
            elif xtype == 'INT':
                xtypes += [ORD, ]
            elif xtype == 'ENUM':
                xtypes += [(ENUM, len(limits)), ]

            xlimits += [limits]

        sampling = MixedIntegerSamplingMethod(xtypes=xtypes, xlimits=xlimits,
                                                sampling_method_class=LHS, criterion="ese",
                                                random_state=random_state)

        input_samples = sampling(n_samples)

        return input_samples

    def _bounded_perf(self,inputs: np.ndarray) -> np.ndarray:
        """
        Retrieve performance estimates from the surrogate

        Parameters
        ----------
        inputs : np.ndarray
            inputs at which to make predictions must be of size
            n_samples x n_margin_nodes

        Returns
        -------
        np.ndarray
            array of estimated performance parameters
        """
        perf_estimate = self.sm_perf.predict_values(inputs)
        perf_estimate = np.clip(perf_estimate,self.lb_outputs,self.ub_outputs)
        return perf_estimate

    def __copy__(self):
        """
        returns a shallow copy of MarginNetwork instance
        https://stackoverflow.com/a/15774013

        Returns
        -------
        MarginNetwork
            shallow copy of MarginNetwork instance
        """
        id_self = id(self) # memoization avoids unnecessary recursion
        return type(self)(self.design_params, self.input_specs,
                 self.fixed_params, self.behaviours, self.decisions,
                 self.margin_nodes, self.performances, self.key+'_copy_'+str(id_self))
    
    def __deepcopy__(self, memo): # memo is a dict of id's to copies
        """
        creates a deep independent copy of the class instance self.
        https://stackoverflow.com/a/15774013

        Parameters
        ----------
        memo : Dict
            memoization dictionary of id(original) (or identity numbers) to copies

        Returns
        -------
        MarginNetwork
            copy of MarginNetwork instance
        """
        id_self = id(self) # memoization avoids unnecessary recursion
        _copy = memo.get(id_self)
        if _copy is None:
            _copy = type(self)(deepcopy(self.design_params,memo), deepcopy(self.input_specs,memo),
                               deepcopy(self.fixed_params,memo), deepcopy(self.behaviours,memo), 
                               deepcopy(self.decisions,memo),deepcopy(self.margin_nodes,memo), 
                               deepcopy(self.performances,memo), self.key)
            _copy.key = self.key+'_copy_'+str(id(_copy))
            memo[id_self] = _copy 
        return _copy


def nearest(p1: np.ndarray, p2: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Find the nearest point to s along a line given by p1 and p2
    https://stackoverflow.com/a/47198877
    https://stackoverflow.com/a/39840218
    https://math.stackexchange.com/questions/13176/how-to-find-a-point-on-a-line-closest-to-another-given-point

    Parameters
    ----------
    p1 : np.ndarray
        first point on the line
    p2 : np.ndarray
        second point on the line
    s : np.ndarray
        point to calculate distance to

    Returns
    -------
    np.ndarray
        coordinates of the nearest point
    """
    x1, y1 = p1
    x2, y2 = p2
    xs, ys = s
    dx, dy = x2 - x1, y2 - y1

    # n = (p2 - p1) / (np.linalg.norm(p1 - p2))
    # n_o = np.array([-n[1], n[0]])

    # qp = s - p1
    # d = np.dot(qp,n_o) # distance
    # s_n = s - np.dot(qp,n_o) * n_o # nearest point

    det = dx * dx + dy * dy
    a = (dy * (ys - y1) + dx * (xs - x1)) / det
    s_n = np.array((x1 + a * dx, y1 + a * dy)) # nearest point
    d = (np.cross(p2 - p1, s - p1)) / np.linalg.norm(p2 - p1) # distance

    return s_n, d

def _sample_man(input_i: np.ndarray, man: MarginNetwork, sampling_freq: int = 1) -> Tuple[np.ndarray,List[Union[int,float]],np.ndarray]:
    """
    samples a given MarginNetwork

    Parameters
    ----------
    input_i : np.ndarray
        Inputs to the set the MarginNetwork before running it
    man : MarginNetwork
        the margin network used for the calculation. Must provide the forward() method
    sampling_freq : int, optional
        if the MarginNetwork is stochastic then set to greater than 1 to take the mean of multiple runs, by default 1

    Returns
    -------
    Tuple[np.ndarray,List[Union[int,float]],np.ndarray]
        List of target threshold values, input specifications, and performance parameters
    """

    design = input_i[:len(man.design_params)]
    spec = input_i[len(man.design_params):len(man.design_params) + len(man.input_specs)]
    decision = input_i[len(man.design_params) + len(man.input_specs):]
    decision_variable_types = man.variable_type[len(man.design_params) + len(man.input_specs):]

    decisions = []
    for value, universe, xtype in zip(decision, man.universe_decision, decision_variable_types):

        if xtype == 'ENUM':
            # CATEGORICAL variables
            decisions += [universe[int(value)]] # for categorical variables an integer corresponding to universe position is passed
        elif xtype == 'INT':
            # ORDINAL variables
            decisions += [value]

    man.design_vector = design  # Set design parameters to their respective values
    man.spec_vector = spec  # Set input specifications to their respective values
    man.decision_vector = decisions  # Set decisions to their respective values
    man.init_decisions() # to accomodate decided values dependant on design parameters

    node_samples = np.empty((0, len(man.margin_nodes)))
    perf_samples = np.empty((0, len(man.performances)))
    for n in range(sampling_freq):
        # man.randomize() # Randomize the MAN
        man.allocate_margins('manual')
        man.forward()  # Run one pass of the MAN

        node_samples = np.vstack((node_samples, man.dv_vector.reshape(1, node_samples.shape[1])))
        perf_samples = np.vstack((perf_samples, man.perf_vector.reshape(1, perf_samples.shape[1])))

    node_samples = np.mean(node_samples, axis=0)
    perf_samples = np.mean(perf_samples, axis=0)

    return node_samples, spec, perf_samples

def _parallel_sample_man(input_i: np.ndarray, mans: List[MarginNetwork], 
                         sampling_freq: int = 1, process_ids: List[int] = None) -> Tuple[np.ndarray,List[Union[int,float]],np.ndarray]:
    """
    samples a given MarginNetwork

    Parameters
    ----------
    input_i : np.ndarray
        Inputs to the set the MarginNetwork before running it
    mans : List[MarginNetwork]
        the margin network(s) used for the calculation. Must provide the forward() method
    sampling_freq : int, optional
        if the MarginNetwork is stochastic then set to greater than 1 to take the mean of multiple runs, by default 1

    Returns
    -------
    Tuple[np.ndarray,List[Union[int,float]],np.ndarray]
        List of target threshold values, input specifications, and performance parameters
    """

    # Select a man to forward based on process id
    if process_ids == None:
        pid = None
        man = mans[0]
    else:
        pid = mp.current_process()._identity[0] - process_ids[0]
        man = mans[pid]

    tt_samples, spec, perf_samples = _sample_man(input_i,man,sampling_freq)

    return tt_samples, spec, perf_samples

def _sample_behaviour(*args, variable_dict = None, **kwargs) -> List[Union[int,float]]:
    """
    samples the decision universe
    *args are expected to vary
    **kwargs are fixed

    Parameters
    ----------
    behviours : List[Behaviour]
        the Behaviour(s) used for the calculation. Must provide the __call__() method that returns outputs of interest
    process_ids : List[int]
        process ids of the current worker pool (used to correct pid)

    Returns
    -------
    List[Union[int,float, List[Union[int,float]]]]
        List of outputs of interest
    """

    # Select a man to forward based on process id
    if kwargs['process_ids'] == None:
        pid = None
        behviour = kwargs['behaviours'][0]
    else:
        pid = mp.current_process()._identity[0] - kwargs['process_ids'][0]
        behviour = kwargs['behaviours'][pid]

    if variable_dict is not None:
        args_b = []
        # convert categorical indices to their values
        for i,value in enumerate(variable_dict.values()):
            if value['type'] == 'ENUM':
                # CATEGORICAL variables
                args_b += [value['limits'][int(args[i])]] # for categorical variables an integer corresponding to universe position is passed
            else:
                args_b += [args[i]]
    else:
        args_b = args

    # only pass this argument if parallel computation is requested (this means the user 
    # does not need to pass **kwargs to decided_value_model)
    if pid is not None:
        behviour(*args_b,id=pid+1)
    else:
        behviour(*args_b)
     
    return behviour._intermediate, behviour._performance, behviour._decided_value, behviour._threshold

def _secant_method(f: callable, a: float, b: float, lb: float = -np.inf, ub: float = np.inf, 
                   eps: float = 1e-12, tol: float=1e-6, max_iter: int=100) -> float:
    """
    Finds a root of the function f using the secant method.

    Parameters
    ----------
    f : callable
        The function to find the root of
    a : float
        The first initial guess
    b : float
        The second initial guess
    lb : float
        The lower bounds for the search
    ub : float
        The upper bounds for the search
    eps : float, optional
        The convergence tolerance. The algorithm stops when the relative
        error between two successive iterates is less than eps. Default is 1e-9
    tol : float, optional
        The desired tolerance for the root
    max_iter : int, optional
        The maximum number of iterations

    Returns
    -------
    float or None
        The estimated root of the function f, or None if the maximum number of iterations is reached
    """

    # Initialize variables
    iter_count = 0

    if abs(f(b) - f(a)) <= tol:
        a += eps

    # Loop until the tolerance is achieved or maximum iterations are reached
    while iter_count < max_iter and abs(a) <= 1e40 and abs(b) <= 1e40:
        c = clamp(b - (f(b) * (b - a + eps)) / (f(b) - f(a) + eps),lb,ub)
        if abs(f(c)) < tol:
            return c
        a = clamp(b,lb,ub)
        b = clamp(c,lb,ub)
        iter_count += 1

    # If the maximum number of iterations is reached, return None
    return None

def _fixed_step_method(f: callable, a: float, alpha: float, delta_max: float = 1e3, max_iter: int=1e4) -> float:
    """
    Finds a root of the function f using the secant method.

    Parameters
    ----------
    f : callable
        The function to find the root of
    a : float
        The first initial guess
    alpha : float
        The step size parameter
    delta_max : float, optional
        The maximum allowable change in the function f
    max_iter : int, optional
        The maximum number of iterations

    Returns
    -------
    float or None
        The estimated root of the function f, or None if the maximum number of iterations is reached
    """

    # Initialize variables
    iter_count = 0

    # Loop until the tolerance is achieved or maximum iterations are reached
    delta = 0.0
    fold = f(a)
    a_final = a
    while iter_count <= max_iter and delta <= delta_max:
        a += alpha
        fnew = f(a)
        delta = abs(fnew - fold)
        fold = fnew
        iter_count += 1

        if fold >= 0:
            a_final = a

        if fold < 0:
            return a_final

    # If the maximum number of iterations is reached, return None
    return None
