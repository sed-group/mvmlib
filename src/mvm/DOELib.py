from typing import List, Union, Optional

import numpy as np
from pyDOE2 import lhs
import os, json

from .utilities import check_folder, serialize

"""DOE Library for generating a design"""


def gridsamp(bounds: np.ndarray, q: Union[int, np.ndarray, List[int]]) -> np.ndarray:
    """
    GRIDSAMP  n-dimensional grid over given range

    Parameters
    ----------
    bounds : np.ndarray
        2*n matrix with lower and upper limits
    q : np.ndarray
        n-vector, q(j) is the number of points
        in the j'th direction.
        If q is a scalar, then all q(j) = q

    Returns
    -------
    S : np.ndarray
        m*n array with points, m = prod(q)
    """

    [mr, n] = np.shape(bounds)
    dr = np.diff(bounds, axis=0)[0]  # difference across rows
    if mr != 2 or any([item < 0 for item in dr]):
        raise Exception('bounds must be an array with two rows and bounds(1,:) <= bounds(2,:)')

    if q.ndim > 1 or any([item <= 0 for item in q]):
        raise Exception('q must be a vector with non-negative elements')

    p = len(q)
    if p == 1:
        q = np.tile(q, (1, n))[0]
    elif p != n:
        raise Exception('length of q must be either 1 or %d' % n)

    # Check for degenerate intervals
    i = np.where(dr == 0)[0]
    if i.size > 0:
        q[i] = 0 * q[i]

    # Recursive computation
    if n > 1:
        a = gridsamp(bounds[:, 1::], q[1::])  # Recursive call
        [m, _] = np.shape(a)
        q = int(q[0])
        s = np.concatenate((np.zeros((m * q, 1)), np.tile(a, (q, 1))), axis=1)
        y = np.linspace(bounds[0, 0], bounds[1, 0], q)

        k = range(m)
        for i in range(q):
            aug = np.tile(y[i], (m, 1))
            aug = np.reshape(aug, s[k, 0].shape)

            s[k, 0] = aug
            k = [item + m for item in k]
    else:
        s = np.linspace(bounds[0, 0], bounds[1, 0], int(q[-1]))
        s = np.transpose([s])

    return s


def scaling(x: np.ndarray, lb: np.ndarray, ub: np.ndarray, operation: int) -> np.ndarray:
    """
    Scaling by a range

    Parameters
    ----------
    x : np.ndarray
        2d array of size n * nsamples of datapoints
    lb : np.ndarray
        1d array of length n specifying lower range of features
    ub : np.ndarray
        1d array of length n = len(l) specifying upper range of features
    operation : int
        The flag type indicates whether to scale (1) or unscale (2)

    Returns
    -------
    x_out : np.ndarray
        2d array of size n * nsamples of unscaled datapoints
    """

    if operation == 1:
        # scale
        x_out = (x - lb) / (ub - lb)
        return x_out
    elif operation == 2:
        # unscale
        x_out = lb + x * (ub - lb)
        return x_out


class Design:

    def __init__(
            self, 
            lb: np.ndarray, 
            ub: np.ndarray, 
            nsamples: Union[int, List[int], np.ndarray], 
            doe_type: str, 
            random_seed: Optional[int] = None
        ):
        """
        Contains the experimental design limits, 
        samples and other relevant statistics

        Parameters
        ----------
        lb : np.ndarray
            1d array of length n specifying lower range of features
        ub : np.ndarray
            1d array of length n = len(lb) specifying upper range of features
        nsamples : Union[int,List[int],np.ndarray]
            The number of samples to generate for each factor, 
            if array_like and type specificed is 'fullfact' then samples each variable according to its sampling vector
        doe_type : str, optional
            Allowable values are "LHS" and "fullfact". If no value 
            given, the design is simply randomized.
        random_seed : str, optional
            random seed for initializing LHS DOE. Does not affect 'fullfact' DOE's
            If no value given then results are not reproducible
        design_matrix : str, optional
            use the provided design_matrix instead of the one generated initially
            
        # TODO: return error if len(lb) != len(ub)
        """

        self._lb = list(lb)
        self._ub = list(ub)

        self.type = doe_type
        self.seed = random_seed

        self._nlevels = []
        self._nsamples = 0
        self._design = np.empty((0,len(self._lb)))

        # Generate latin hypercube design and store it

        if self.type == 'LHS':
            assert type(nsamples) == int
            self._nsamples = nsamples
            self._design = lhs(len(self._lb), samples=self._nsamples, random_state=self.seed)
        elif self.type == 'fullfact':
            bounds = np.array([[0.0] * len(self._lb), [1.0] * len(self._ub)])
            if type(nsamples) == list:
                assert len(nsamples) == len(self._lb)
                self._nlevels = nsamples
                self._nsamples = np.prod(nsamples)
                self._design = gridsamp(bounds, np.array(self._nlevels))
            if type(nsamples) == np.ndarray:
                assert nsamples.ndim == 1
                assert len(nsamples) == len(self._lb)
                self._nlevels = nsamples.tolist()
                self._nsamples = np.prod(nsamples)
                self._design = gridsamp(bounds, np.array(self._nlevels))
            elif type(nsamples) == int:
                n_levels_list = [nsamples,]*len(self._lb)
                diff = np.array(self._lb) - np.array(self._ub)
                n_levels_array = np.array(n_levels_list,dtype=int)
                n_levels_array[diff==0] = 0
                self._nlevels = n_levels_array.tolist()

                self._nsamples = np.prod(self._nlevels)
                self._design = gridsamp(bounds, np.array(self._nlevels))

    @property
    def design(self) -> np.ndarray:
        """
        used to return a copy of the design matrix

        Returns
        -------
        np.ndarray
            copy of design matrix
        """
        return self._design.copy()

    @design.setter
    def design(self,value:np.ndarray) -> None:
        """
        used to set the design matrix externally

        Parameters
        ----------
        value : np.ndarray
            numpy array of the matrix
        """

        n_samples = np.prod(self._nlevels) if self.type == "fullfact" else self._nsamples

        assert value.shape[0] == self._nsamples
        assert value.shape[1] == len(self._lb)

        self._design = value

    def unscale(self) -> np.ndarray:
        """
        Unscale latin hypercube by ub and lb

        Returns
        -------
        unscaled_LH : np.array
            numpy array of size n * nsamples of LH values unscaled by lb and ub
        """

        unscaled_lh = scaling(self._design, np.array(self._lb), np.array(self._ub), 2)

        return unscaled_lh

    def scale(self) -> np.ndarray:
        """
        return scaled latin hypercube between 0 and 1

        Returns
        -------
        scaled_LH : np.array
            numpy array of size n * nsamples of LH values between 0 and 1
        """

        return self._design.copy()

    def save(self,name:str) -> None:
        """
        saves the state of the DOE to a text file

        Parameters
        ----------
        file : str
            the folder name to be used to save the data which includes
            * settings.json
            * scale.csv
            * unscale.csv
        """

        exists = check_folder(name)

        if exists:
            print("warning folder %s already exists!" %name)

        with open(os.path.join(name,"scale.csv"),"w") as f:
            np.savetxt(f,self._design)

        with open(os.path.join(name,"unscale.csv"),"w") as f:
            np.savetxt(f,self.unscale())

        with open(os.path.join(name,"settings.json"),"w") as f:
            
            settings = {
                "n_samples" : self._nsamples,
                "n_levels" : self._nlevels,
                "lb" : self._lb,
                "ub" : self._ub,
                "type" : self.type,
                "random_seed" : self.seed
            }
            json.dump(serialize(settings), f, indent=4)
            print(json.dumps(serialize(settings), indent=4))

    def load(self,name:str) -> None:
        """
        loads the state of the DOE from a text file

        Parameters
        ----------
        file : str
            the folder name to be used to load the data which includes
            * settings.json
            * scale.csv
            * unscale.csv
        """

        exists = check_folder(name)
        assert exists, "directory %s does not exist!" %name
        assert os.path.isfile(os.path.join(name,"scale.csv")), "file %s/%s does not exist!" %(name,"scale.csv")
        assert os.path.isfile(os.path.join(name,"unscale.csv")), "file %s/%s does not exist!" %(name,"unscale.csv")
        assert os.path.isfile(os.path.join(name,"settings.json")), "file %s/%s does not exist!" %(name,"settings.json")

        with open(os.path.join(name,"scale.csv"),"r") as f:
            self._design = np.loadtxt(f)

        with open(os.path.join(name,"settings.json"),"r") as f:
            settings = json.load(f)

        print(json.dumps(settings, indent=4))

        self.type = settings["type"]
        self._nlevels = settings["n_samples"]
        self._nsamples = settings["n_levels"]
        self._lb = settings["lb"]
        self._ub = settings["ub"]
        self.type = settings["type"]
        self.seed = settings["random_seed"]

def get_design(name:str) -> Design:
    """
    returns an a Design object initialized using the input folder

    Parameters
    ----------
    file : str
        the folder name to be used to load the data which includes
        * settings.json
        * scale.csv
        * unscale.csv
    """

    exists = check_folder(name)
    assert exists, "directory %s does not exist!" %name
    assert os.path.isfile(os.path.join(name,"scale.csv")), "file %s/%s does not exist!" %(name,"scale.csv")
    assert os.path.isfile(os.path.join(name,"unscale.csv")), "file %s/%s does not exist!" %(name,"unscale.csv")
    assert os.path.isfile(os.path.join(name,"settings.json")), "file %s/%s does not exist!" %(name,"settings.json")

    with open(os.path.join(name,"scale.csv"),"r") as f:
        design = np.loadtxt(f)

    with open(os.path.join(name,"settings.json"),"r") as f:
        settings = json.load(f)

    print(json.dumps(settings, indent=4))

    if settings["type"] == "LHS":

        doe = Design(
            lb=np.array(settings["lb"]),
            ub=np.array(settings["ub"]),
            nsamples=settings["n_samples"],
            doe_type =settings["type"],
            random_seed=settings["random_seed"]
        )

    elif settings["type"] == "fullfact":

        doe = Design(
            lb=np.array(settings["lb"]),
            ub=np.array(settings["ub"]),
            nsamples=settings["n_levels"],
            doe_type =settings["type"],
            random_seed=settings["random_seed"]
        )

    doe.design = design

    return doe

