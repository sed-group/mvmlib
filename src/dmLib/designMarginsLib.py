from abc import ABC, abstractmethod
from typing import Tuple, List, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from smt.surrogate_models import KRG

from .DOELib import Design, scaling
from .uncertaintyLib import Distribution, GaussianFunc, UniformFunc, VisualizeDist
from .utilities import check_folder

"""Design margins library for computing buffer and excess"""


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
    det = dx * dx + dy * dy
    a = (dy * (ys - y1) + dx * (xs - x1)) / det

    # calculate distance
    d = (np.cross(p2 - p1, s - p1)) / np.linalg.norm(p2 - p1)

    return np.array((x1 + a * dx, y1 + a * dy)), d


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
            assert n <= len(self._values)
            self._values = self._values[..., :-n].copy()  # select along last dimension
            self.value = self.values[..., -1].copy()
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


class ScalarParam(Cache):
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
        super().__init__(key, dims)

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
        dmLib.Distribution
            instance of dmLib.Distribution holding value pdf
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
        value_hist = np.histogram(values, bins=50, density=True)
        self._value_dist = Distribution(value_hist[0], lb=min(value_hist[1]), ub=max(value_hist[1]))

    def view(self, xlabel: str = None, folder: str = '', file: str = None, img_format: str = 'pdf'):
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
        """
        if xlabel is None:
            xlabel = '%s' % self.key

        vis = VisualizeDist(values=self.values)
        vis.view(xlabel=xlabel, folder=self.key, file=file, img_format=img_format)

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
        if xlabel is None:
            xlabel = '%s' % self.key

        vis = VisualizeDist(values=self.values)
        vis.view_cdf(xlabel=xlabel, folder=self.key, file=file, img_format=img_format)

    def reset(self, n: int):
        """
        Resets accumulated random observations and value distributions

        Parameters
        -----------
        n : int, optional
            if provided deletes only the last n_samples, by default None
        """

        super().reset(n)
        if n is not None:
            self.value_dist = self.values
        else:
            self._value_dist = None

    def __call__(self, value: Union[float, np.ndarray]):
        """
        Set the value of the parameter

        Parameters
        ----------
        value : Union[float,np.ndarray]
            values of the parameter.
            The length of this vector equals the number of samples
        """

        super().__call__(value)
        self.value_dist = self.values


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
    def build_param(key: str, dims: List[int]) -> Union[ScalarParam, VectorParam, MatrixParam]:
        """
        Returns appropriate parameter class based on supplied dimensions

        Parameters
        ----------
        key : str
            unique string identifier
        dims : List[int]
            The dimensions of the parameter

        Returns
        -------
        Union[ScalarParam,VectorParam,MatrixParam]
            an instance of the correct class
        """
        if len(dims) == 0:
            return ScalarParam(key, dims)
        elif len(dims) == 1:
            return VectorParam(key, dims)
        elif len(dims) == 2:
            return MatrixParam(key, dims)


class FixedParam:
    def __init__(self, value: Union[float, int, str], key: str,
                 description: str = '', symbol: str = ''):
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

        self.description = description
        self.symbol = symbol
        self.key = key
        self.type = type(value)
        self.value = value


class DesignParam:
    def __init__(self, value: Union[float, int, str], key: str,
                 universe: Union[Tuple[Union[int, float], Union[int, float]], List[Union[int, float]]],
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
        universe : Union[Tuple[Union[int,float],Union[int,float]],List[Union[int,float]]]
            the possible values the design parameter can take, 
            If tuple must be of length 2 (upper and lower bound)
            type(value) must be float, or int
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

        if type(universe) == tuple:
            assert len(universe) == 2
            assert self.type in [float, int]
        elif type(universe) == list:
            assert len(universe) > 0
        self.original = value
        self.universe = universe

    def reset(self):
        """
        resets the design parameters to its initial value given at `__init__`   
        """
        self.value = self.original


class InputSpec(ScalarParam):
    def __init__(self, value: Union[float, int, Distribution, GaussianFunc, UniformFunc],
                 key: str, universe: Tuple[float, float], description: str = '', symbol: str = '',
                 distribution: Distribution = None,
                 cov_index: int = 0, inc: float = 5.0, inc_type: str = 'rel'):
        """
        Contains description of an input specification
        could deterministic or stochastic

        Parameters
        ----------
        value : Union[float,int,Distribution,GaussianFunc]
            the value of the input spec, 
            if type is Distribution then a sample is drawn
        key : str
            unique identifier
        universe : Tuple[float,float]
            the possible values the design parameter can take, 
            must be of length 2 (upper and lower bound)
            type(value) must be float
        description : str, optional
            description string, by default ''
        symbol : str, optional
            shorthand symbol, by default ''
        distribution : Distribution, optional
            if a Distribution object is provided, then the spec can sampled by calling it
            Example:
            >>> from dmLib import InputSpec, GaussianFunc
            >>> dist = GaussianFunc(1.0, 0.1)
            >>> s1 = InputSpec(1.0, 'S1', (0.0, 1.0), distribution=dist)
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

        self.description = description
        self.symbol = symbol
        self.distribution = distribution
        self.cov_index = cov_index
        self.inc = inc
        self.inc_type = inc_type
        self.original = value
        self.universe = universe

        self.value = value

        # Check if input spec is stochastic
        if type(distribution) in [Distribution, GaussianFunc, UniformFunc]:
            self.stochastic = True
        else:
            self.stochastic = False

        # Check if input spec is co-dependant on another
        if type(value) == GaussianFunc:
            self.ndim = value.ndim
        else:
            self.ndim = 1

        assert self.cov_index <= self.ndim

    def reset(self, n: int = None):
        """
        Resets accumulated random observations and value distributions

        Parameters
        -----------
        n : int, optional
            if provided deletes only the last n_samples, by default None
        """
        super().reset(n)
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

        if self.stochastic:
            assert self.distribution.samples.shape[1] >= n
            sampled_values = self.distribution.samples[self.cov_index, -n:]  # retrieve last N samples
            self.value = self.distribution.samples[self.cov_index, -1]  # retrieve last sample from distribution
            self.original = self.distribution.samples[self.cov_index, -1]  # retrieve last sample from distribution
        else:
            sampled_values = self.original * np.ones(n)

        self.values = sampled_values
        return sampled_values  # return the requested number of samples


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


class Behaviour(ABC):
    def __init__(self, key: str = ''):
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
        self.key = key
        self.intermediate = None
        self.performance = None
        self.decided_value = None
        self.threshold = None

    def reset(self):
        """
        Resets the stored variables
        """
        self.intermediate = None
        self.performance = None
        self.decided_value = None
        self.threshold = None

    @abstractmethod
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
        >>> class MyBehaviour(Behaviour):
        >>>     def __call__(self,r,d):
        >>>         # some specific model-dependent behaviour
        >>>         self.intermediate = d
        >>>         self.performance = r*2+1 / d
        >>>         self.decided_value = r**2
        >>>         self.threshold = r/d
        """
        # default code for the default behaviour
        return


class MarginNetwork(ABC):
    def __init__(self, design_params: List[DesignParam], input_specs: List[InputSpec],
                 fixed_params: List[FixedParam], behaviours: List[Behaviour],
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
        margin_nodes : List[MarginNode]
            list of MarginNode instances
        margin_nodes : List[Performance]
            list of Performance instances
        key : str, optional
            string to tag instance with, default = ''
        """
        self.sm_perf = None
        self.fig = None
        self.lb_inputs = None
        self.ub_inputs = None
        self.key = key

        # Inputs
        self.design_params = design_params
        self.input_specs = input_specs
        self.fixed_params = fixed_params
        self.behaviours = behaviours

        # Outputs
        self.margin_nodes = margin_nodes
        self.performances = performances
        self.deterioration_vector = ParamFactory.build_param(key=self.key, dims=[len(input_specs)])
        self.impact_matrix = ParamFactory.build_param(key=self.key, dims=[len(margin_nodes), len(performances), ])
        self.absorption_matrix = ParamFactory.build_param(key=self.key, dims=[len(margin_nodes), len(input_specs), ])
        self.utilization_matrix = ParamFactory.build_param(key=self.key, dims=[len(margin_nodes), len(input_specs), ])

        # Design parameter space
        lb = np.array([])
        ub = np.array([])
        # Get upper and lower bound for continuous variables
        for design_param in self.design_params:
            assert min(design_param.universe) < max(design_param.universe), \
                'max of universe of design parameter %s must be greater than min' % design_param.key

            lb = np.append(lb, min(design_param.universe))
            ub = np.append(ub, max(design_param.universe))

        self.lb_d, self.ub_d = lb, ub

        # Input specification space
        lb = np.array([])
        ub = np.array([])
        for input_spec in self.input_specs:
            assert min(input_spec.universe) < max(input_spec.universe), \
                'max of universe of input spec %s must be greater than min' % input_spec.key

            lb = np.append(lb, min(input_spec.universe))
            ub = np.append(ub, max(input_spec.universe))

        self.lb_s, self.ub_s = lb, ub

        self.lb_i = np.append(self.lb_d, self.lb_s)
        self.ub_i = np.append(self.ub_d, self.ub_s)

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

    def train_performance_surrogate(self, n_samples: int = 100, sampling_freq: int = 1,
                                    bandwidth: List[float] = [1e-2],
                                    ext_samples: Tuple[np.ndarray, np.ndarray] = None):
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
        ext_samples : tuple[np.ndarray,np.ndarray], optional
            if sample data provided externally then use directly to fit the response surface, 
            Tuple must have length 2, ext_samples[0] must have shape (N_samples,len(margin_nodes)),
            ext_samples[1] must have shape (N_samples,len(performances)),
            by default None
        """

        if ext_samples is None:
            # generate training data for response surface using an LHS grid of design parameter and input
            # specification space
            input_samples = Design(self.lb_i, self.ub_i, n_samples, "LHS").unscale()  # 2D grid

            xt = np.empty((0, len(self.margin_nodes) + len(self.input_specs)))  # excess + input secs
            yt = np.empty((0, len(self.performances)))  # Performance parameters

            for input_i in input_samples:
                design = input_i[:len(self.design_params)]
                spec = input_i[len(self.design_params):]

                self.design_vector = design  # Set design parameters to their respective values
                self.spec_vector = spec  # Set input specifications to their respective values

                excess_samples = np.empty((0, len(self.margin_nodes)))
                perf_samples = np.empty((0, len(self.performances)))
                for n in range(sampling_freq):
                    # self.randomize() # Randomize the MAN
                    self.forward()  # Run one pass of the MAN

                    excess_samples = np.vstack((excess_samples, self.excess_vector.reshape(1, excess_samples.shape[1])))
                    perf_samples = np.vstack((perf_samples, self.perf_vector.reshape(1, perf_samples.shape[1])))

                excess_samples = np.mean(excess_samples, axis=0)
                perf_samples = np.mean(perf_samples, axis=0)

                # concatenate input specifications
                x_i = np.append(excess_samples, spec)
                x_i = x_i.reshape(1, xt.shape[1])

                xt = np.vstack((xt, x_i))
                yt = np.vstack((yt, perf_samples.reshape(1, yt.shape[1])))

        else:
            assert type(ext_samples) == tuple
            assert len(ext_samples) == 2
            for value in ext_samples:
                assert type(value) == np.ndarray
            assert ext_samples[0].shape[1] == len(self.margin_nodes) + len(self.input_specs)
            assert ext_samples[1].shape[1] == len(self.performances)
            assert ext_samples[0].shape[0] == ext_samples[1].shape[0]

            xt = ext_samples[0]
            yt = ext_samples[1]

        # Get lower and upper bounds of excess values
        self.lb_inputs = np.min(xt, axis=0)
        self.ub_inputs = np.max(xt, axis=0)
        xt = scaling(xt, self.lb_inputs, self.ub_inputs, operation=1)

        self.reset()
        assert len(bandwidth) == 1 or len(bandwidth) == xt.shape[1], \
            'bandwidth list size must be at least %i or 1' % xt.shape[1]

        self.sm_perf = KRG(theta0=bandwidth, print_prediction=False)
        self.sm_perf.set_training_values(xt, yt)
        self.sm_perf.train()

    def view_perf(self, e_indices: List[int], p_index: int, label_1: str = None, label_2: str = None,
                  label_p: str = None, n_levels: int = 100, folder: str = '', file: str = None,
                  img_format: str = 'pdf'):
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

        lb_excess, ub_excess = np.empty(0), np.empty(0)
        for margin_node in self.margin_nodes:
            lb_excess = np.append(lb_excess, margin_node.excess.value)
            ub_excess = np.append(ub_excess, margin_node.excess.value + 1e-3)
        lb_excess[e_indices] = self.lb_inputs[e_indices]
        ub_excess[e_indices] = self.ub_inputs[e_indices]

        excess_doe = Design(lb_excess, ub_excess, sampling_vector, 'fullfact')
        values = np.hstack((excess_doe.unscale(),
                            np.tile(self.nominal_spec_vector, (excess_doe.unscale().shape[0], 1))))
        values = scaling(values, self.lb_inputs, self.ub_inputs, operation=1)

        perf_estimate = self.sm_perf.predict_values(values)

        x = excess_doe.unscale()[:, e_indices[0]].reshape((n_levels, n_levels))
        y = excess_doe.unscale()[:, e_indices[1]].reshape((n_levels, n_levels))
        z = perf_estimate[:, p_index].reshape((n_levels, n_levels))

        if label_1 is None:
            label_1 = 'E_%s' % self.margin_nodes[e_indices[0]].key
        if label_2 is None:
            label_2 = 'E_%s' % self.margin_nodes[e_indices[1]].key
        if label_p is None:
            label_p = self.performances[p_index].key

        ax.contourf(x, y, z, cmap=plt.cm.jet, )
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

        excess_values = self.excess_vector.reshape(1, -1)  # turn it into a 2D matrix
        input_specs = self.spec_vector.reshape(1, -1)  # turn it into a 2D matrix

        # Compute performances at decided values first
        if use_estimate:
            # Use surrogate model
            value = np.hstack((excess_values, input_specs))
            value = scaling(value, self.lb_inputs, self.ub_inputs, operation=1)
            performances = self.sm_perf.predict_values(value)
        else:
            # Get performances from behaviour models
            performances = self.perf_vector

        performances = np.tile(performances, (len(self.margin_nodes), 1))

        # performances = [len(margin_nodes), len(performances)]

        # Compute performances at target threshold for each margin node
        input_excess = np.tile(excess_values, (len(self.margin_nodes), 1))  # Create a square matrix

        # input_excess = [len(margin_nodes), len(margin_nodes)]

        np.fill_diagonal(input_excess, np.zeros(len(self.margin_nodes)), wrap=False)  # works in place

        # input_excess = [len(margin_nodes), len(margin_nodes)]

        # concatenate input specifications
        values = np.hstack((input_excess,
                            np.tile(self.nominal_spec_vector, (input_excess.shape[0], 1))))
        values = scaling(values, self.lb_inputs, self.ub_inputs, operation=1)

        thresh_perf = self.sm_perf.predict_values(values)

        # thresh_perf = [len(margin_nodes), len(performances)]

        # Get the sign of the performance parameters
        signs = np.empty(0)
        for performance in self.performances:
            if performance.direction == 'less_is_better':
                sign = 1.0
            elif performance.direction == 'more_is_better':
                sign = -1.0
            else:
                sign = 0.0
                Exception('Performance direction : %s is invalid' % str(performance.direction))
            signs = np.append(signs, sign)

        impact = signs * (performances - thresh_perf) / thresh_perf

        # impact = [len(margin_nodes), len(performances)]

        self.impact_matrix(impact)  # Store impact matrix

    def compute_absorption(self):
        """
        Computes the change absorption capability (CAC) matrix of the MVM that has a size
        [n_margins, n_input_specs] and appends its absorption matrix if called
        multiple times without a reset(). 
        Also stores and appends the deterioration vectors
        """

        # Deterioration computation

        spec_limit = np.empty(0)
        signs = np.empty(0)
        for spec in self.input_specs:

            self.forward()  # do not randomize the man for deterioration
            n_inc = 1
            delta_e = 1.0

            if spec.inc_type == 'rel':
                inc = (spec.original * spec.inc) / 100
            elif spec.inc_type == 'abs':
                inc = spec.inc
            else:
                inc = spec.inc
                Warning('increment type %s, is invalid using absolute value' % str(spec.inc))

            while all(self.excess_vector >= 0) and delta_e <= 1e3 and n_inc <= 1e4:
                excess_last_inc = self.excess_vector

                spec.value += inc
                self.forward()  # do not randomize the man for deterioration
                n_inc += 1

                delta_e = np.min(self.excess_vector - excess_last_inc)

            spec_limit = np.append(spec_limit, spec.value)
            signs = np.append(signs, np.sign(inc))
            self.reset(n_inc)

        deterioration = np.max((signs * (spec_limit - self.nominal_spec_vector) / self.nominal_spec_vector,
                                np.zeros(len(self.input_specs))), axis=0)
        deterioration[deterioration == 0] = np.nan  # replace with nans for division

        # Absorption computation

        nominal_threshold = np.reshape(self.tt_vector, (len(self.margin_nodes), -1))
        target_thresholds = np.tile(nominal_threshold, (1, len(self.input_specs)))

        # target_thresholds = [len(margin_nodes), len(input_specs)]

        deterioration_matrix = np.tile(deterioration, (len(self.margin_nodes), 1))

        # deterioration_matrix = [len(margin_nodes), len(input_specs)]

        # Compute performances at the spec limit for each margin node

        new_thresholds = np.empty((len(self.margin_nodes), 0))
        for input_spec, input_spec_limit in zip(self.input_specs, spec_limit):
            input_spec.value = input_spec_limit
            self.forward()

            new_thresholds_vector = np.reshape(self.tt_vector, (len(self.margin_nodes), -1))
            new_thresholds = np.hstack((new_thresholds, new_thresholds_vector))

            self.reset(1)

        absorption = np.maximum(abs(new_thresholds - target_thresholds) / (target_thresholds * deterioration_matrix),
                                np.zeros_like(new_thresholds))
        absorption[absorption == np.nan] = 0  # replace undefined absorptions with 0

        # absorption = [len(margin_nodes), len(input_specs)]

        # utilization computation

        decided_value = np.reshape(self.dv_vector, (len(self.margin_nodes), -1))
        decided_values = np.tile(decided_value, (1, len(self.input_specs)))

        # decided_values = [len(margin_nodes), len(input_specs)]

        utilization = 1 - ((decided_values - new_thresholds) / (decided_values - target_thresholds))

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

            ax.scatter(x, y, s=50, c=c)

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
        >>> from dmLib import MarginNetwork, InputSpec, GuassianFunc
        >>> dist_1 = GuassianFunc(1.0,0.1)
        >>> dist_2 = GuassianFunc(0.5,0.2)
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
        >>> from dmLib import MarginNetwork
        >>> class MyMarginNetwork(MarginNetwork):
        >>>     def randomize(self):
        >>>         pass
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

        for design_param in self.design_params:
            design_param.reset()
        for input_spec in self.input_specs:
            input_spec.reset()
        for behaviour in self.behaviours:
            behaviour.reset()
        for margin_node in self.margin_nodes:
            margin_node.reset(n)
        for performance in self.performances:
            performance.reset(n)

    def reset_outputs(self, n: int = None):
        """
        Resets Impact, Absorption, and Deterioration matrices

        Parameters
        -----------
        n : int, optional
            if provided deletes only the last n_samples, by default None
        """

        self.impact_matrix.reset(n)
        self.absorption_matrix.reset(n)
