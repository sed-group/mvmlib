import pytest
import numpy as np
import scipy.stats as stats
from mvm import Design
from mvm import get_design

seed = 1234

def test_fullfactorial():
    """testing full factorial design"""

    lb = np.array([0.0, 0.0,])
    ub = np.array([6.0, 4.0,])

    n_levels = 3

    output = np.array([
        [0.0, 0.0,],
        [0.0, 2.0,],
        [0.0, 4.0,],
        [3.0, 0.0,],
        [3.0, 2.0,],
        [3.0, 4.0,],
        [6.0, 0.0,],
        [6.0, 2.0,],
        [6.0, 4.0,]
        ])

    test = Design(lb,ub,n_levels,'fullfact').unscale()

    check = test == output
    assert check.all()

    n_levels = [2,3,]

    output = np.array([
        [0.0, 0.0,],
        [0.0, 2.0,],
        [0.0, 4.0,],
        [6.0, 0.0,],
        [6.0, 2.0,],
        [6.0, 4.0,],
        ])

    test = Design(lb,ub,n_levels,'fullfact').unscale()

    check = test == output
    assert check.all()

    n_levels = np.array([2,3,])

    output = np.array([
        [0.0, 0.0,],
        [0.0, 2.0,],
        [0.0, 4.0,],
        [6.0, 0.0,],
        [6.0, 2.0,],
        [6.0, 4.0,],
        ])

    test = Design(lb,ub,n_levels,'fullfact').unscale()

    check = test == output
    assert check.all()

def test_LHS():
    """testing full factorial design"""

    lb = np.array([0.0, 0.1,])
    ub = np.array([6.0, 4.0,])

    n_samples = 1000

    test = Design(lb,ub,n_samples,'LHS',random_seed=seed).unscale()

    # Check each dimension separately
    for dimension in range(test.shape[1]):
        # Normalize the data to a 0-1 range
        data = (test[:, dimension] - lb[dimension]) / (ub[dimension] - lb[dimension])
        # Perform the Kolmogorov-Smirnov test against a uniform distribution
        d, p = stats.kstest(data, 'uniform')
        # Assert that the test passes with a significance level of 0.05
        assert abs(max(data) - 1.0) <= 1e-1
        assert abs(min(data) - 0.0) <= 1e-1
        assert p > 0.05, f"Dimension {dimension+1} does not pass the Kolmogorov-Smirnov test for a uniform distribution"

def test_save_load(tmp_path):
    """test the save/load functionality"""
    directory = tmp_path / "test_save"
    directory.mkdir()

    lb = np.array([0.0, 0.1,])
    ub = np.array([6.0, 4.0,])

    n_samples = 1000

    test_doe = Design(lb,ub,n_samples,'LHS',random_seed=seed)
    test_doe.save(directory)
    test_doe.load(directory)

    loaded_doe = get_design(directory)

    assert loaded_doe.type == test_doe.type
    assert np.allclose(loaded_doe.design, test_doe.design, rtol=1e-6)
    assert np.allclose(loaded_doe._lb, test_doe._lb, rtol=1e-6)
    assert np.allclose(loaded_doe._ub, test_doe._ub, rtol=1e-6)
    assert np.allclose(loaded_doe.seed, test_doe.seed, rtol=1e-6)

def test_save_load_fullfact(tmp_path):
    """test the save/load functionality of full fact class"""
    directory = tmp_path / "test_save"
    directory.mkdir()

    lb = np.array([0.0, 0.1,])
    ub = np.array([6.0, 4.0,])

    n_samples = 10

    test_doe = Design(lb,ub,n_samples,'fullfact',random_seed=1234)
    test_doe.save(directory)
    test_doe.load(directory)

    loaded_doe = get_design(directory)

    assert loaded_doe.type == test_doe.type
    assert np.allclose(loaded_doe.design, test_doe.design, rtol=1e-6)
    assert np.allclose(loaded_doe._lb, test_doe._lb, rtol=1e-6)
    assert np.allclose(loaded_doe._ub, test_doe._ub, rtol=1e-6)
    assert np.allclose(loaded_doe.seed, test_doe.seed, rtol=1e-6)