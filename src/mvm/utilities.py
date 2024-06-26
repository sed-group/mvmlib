""" Useful utility functions """

import os
from multiprocess import Pool
from itertools import repeat
from contextlib import contextmanager
import sys
import inspect
import numpy as np

from typing import List, Any, Callable, Dict, Union, Tuple

def check_folder(folder='render/'):
    """
    check if folder exists, make if not present

    Parameters
    ----------
    folder : str, optional
        name of directory to check, by default 'render/'

    Returns
    -------
    bool
        whether the folder already exists or not
    
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
        return False
    else:
        return True

def starmap_with_kwargs(pool, fn, args_iter, kwargs_iter):
    """
    https://stackoverflow.com/questions/45718523/pass-kwargs-to-starmap-while-using-pool-in-python
    """
    args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
    return pool.starmap(apply_args_and_kwargs, args_for_starmap)

def apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)

def parallel_sampling(func:Callable,
    vargs_iterator:List[List[Any]]=[]   ,vkwargs_iterator:List[Dict[str, Any]]=None,
    fargs:List[Any]=[]                  ,fkwargs:Dict[str, Any]=None,
    num_threads:int=1):
    """
    function used to run parallel computations

    Parameters
    ----------
    func : Callable
        function to parallelize
    vargs_iterator : List[List[Any]], optional
        iterator of arguments, by default []
    vkwargs_iterator : List[Dict[str, Any]], optional
        iterator of keyword arguments, by default None
    fargs : List[Any], optional
        fixed arguments, by default []
    fkwargs : Dict[str, Any], optional
        fixed keyword arguments, by default None
    num_threads : int, optional
        number of parallel threads, by default 1

    Returns
    -------
    _type_
        _description_
    """
    args_iter = []
    for vargs in vargs_iterator:
        args_iter += [[*vargs,*fargs]]

    kwargs_iter = []
    for vkwargs in vkwargs_iterator:
        fkwargs.update(vkwargs)
        fkwargs['process_ids'] = None
        kwargs_iter += [fkwargs.copy()]

    if num_threads > 1:
        pool = Pool(num_threads)

        process_ids = [_curr_process._identity[0] for _curr_process in pool._pool[:]] # get process ids

        for vkwargs in kwargs_iter:
            vkwargs['process_ids'] = process_ids # pass to function to parallelize

        results = starmap_with_kwargs(pool, func, args_iter, kwargs_iter)
        pool.terminate()

        """
        https://stackoverflow.com/a/26005535
        # Loop, terminate, and remove from the process list
        # Use a copy [:] of the list to remove items correctly
        for _curr_process in pool._pool[:]:
            print("Terminating process "+ str(_curr_process.pid))
            _curr_process.terminate()
            pool._pool.remove(_curr_process)
        """

    else:
        results = []
        for args,kwargs in zip(args_iter,kwargs_iter):
            result = func(*args, **kwargs)
            results.append(result)

    return results

def clamp(n, smallest, largest): return max(smallest, min(n, largest))

def serialize(input:Union[list,np.ndarray,dict,type]):
    """
    Converts a dict or list with non-serializable objects to strings

    Parameters
    ----------
    input : dict, list
        input to serialize

    Returns
    -------
    dict, list
        string formatted
    """
    if isinstance(input, dict):
        return {serialize(key): serialize(value)
                for key, value in input.items()}
    elif isinstance(input, (list,tuple,np.ndarray)):
        return [serialize(element) for element in input]
    elif isinstance(input,np.integer):
        return int(input)
    elif isinstance(input,(np.float16, np.float32, np.float64)):
        return float(input)
    elif inspect.isclass(input):
        return input.__name__
    else:
        return input
    
def deserialize(input:Union[list,np.ndarray,dict,type]):
    """
    Converts a dict or list with non-serializable objects to strings

    Parameters
    ----------
    input : dict, list
        input to serialize

    Returns
    -------
    dict, list
        string formatted
    """
    if isinstance(input, dict):
        return {deserialize(key): deserialize(value)
                for key, value in input.items()}
    elif isinstance(input, (list,tuple,np.ndarray)):
        return [deserialize(element) for element in input]
    elif isinstance(input, str):
        return input
    else:
        return input
    
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout