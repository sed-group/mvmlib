import pytest
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from typing import List, Tuple

from dmLib import Design, gaussianFunc, Distribution, MarginNode

# Input set 1
@pytest.fixture
def stochastic_inputs() -> Tuple[gaussianFunc,gaussianFunc]:
    # Target threshold
    mu = np.array([4.0,])
    Sigma = np.array([[0.3**2,],])
    threshold = gaussianFunc(mu, Sigma, 'T1')

    # decided value (capability)
    mu = np.array([4.6,])
    Sigma = np.array([[0.3**2,],])
    decided_value = gaussianFunc(mu, Sigma, 'B1')

    return threshold, decided_value

@pytest.fixture
def deterministic_inputs():
    # Target threshold
    threshold = 4.0

    # decided value (capability)
    decided_value = 5.8

    return threshold, decided_value

@pytest.fixture
def requirement():
    # Requirement specifications
    mu = np.array([370,580])
    Sigma = np.array([
        [50, 25],
        [75, 100],
        ])
    requirement = gaussianFunc(mu, Sigma, 'R1')

    return requirement

def test_deterministic(deterministic_inputs):
    """
    Tests the MarginNode excess calculation method for deterministic threshold and behaviour
    """

    # # DEBUG:
    # # Target threshold
    # threshold = 4.0

    # # decided value (capability)
    # decided_value = 5.8

    # example_mean_std = (threshold, decided_value)

    ######################################################
    # Defining a MarginNode object

    threshold, decided_value = deterministic_inputs

    ######################################################
    # Check excess calculation for one sample
    ThermalNode = MarginNode('EM1')
    ThermalNode(decided_value,threshold)
    assert ThermalNode.excess == np.array([decided_value-threshold])

    ######################################################
    # Check return for multiple inputs
    ThermalNode.reset()
    ThermalNode(np.ones(10)*decided_value,np.ones(10)*threshold)
    assert (ThermalNode.excess == np.ones(10) * (decided_value-threshold)).all()

def test_stochastic(stochastic_inputs):
    """
    Tests the MarginNode excess calculation method for stochastic threshold and behaviour
    """

    # DEBUG:
    # Target threshold
    mu = np.array([4.0,])
    Sigma = np.array([[0.3**2,],])
    threshold = gaussianFunc(mu, Sigma, 'T1')

    # decided value (capability)
    mu = np.array([4.6,])
    Sigma = np.array([[0.3**2,],])
    decided_value = gaussianFunc(mu, Sigma, 'B1')

    stochastic_inputs = (threshold, decided_value)

    ######################################################

    # Defining a MarginNode object
    threshold, decided_value = stochastic_inputs
    decided_value.samples
    
    ######################################################
    # Check excess calculation for one sample
    ThermalNode = MarginNode('EM1')
    ThermalNode(decided_value(),threshold())
    assert ThermalNode.excess == decided_value.samples-threshold.samples

    ######################################################
    # Check sampling accuracy of mean and standard deviaction of excess

    ThermalNode.reset()
    decided_value.reset()
    threshold.reset()

    mu_excess = decided_value.mu - threshold.mu # calculate composite random variable mean
    Sigma_excess = decided_value.Sigma + (((-1)**2) * threshold.Sigma) # calculate composite random variable variance

    ThermalNode(decided_value(10000),threshold(10000))
    
    # Check that means and variances of excess
    assert np.math.isclose(np.mean(ThermalNode.excess), mu_excess.squeeze(), rel_tol=1e-1)
    assert np.math.isclose(np.var(ThermalNode.excess), Sigma_excess.squeeze(), rel_tol=1e-1)

    ######################################################
    # Check that CDF computation is correct
    bin_centers, cdf, excess_limit, reliability = ThermalNode.compute_cdf(bins=500)

    test_excess_pdf = norm(loc=mu_excess,scale=np.sqrt(Sigma_excess))
    test_reliability = 1 - test_excess_pdf.cdf(0).squeeze() 
    test_excess_limit = test_excess_pdf.ppf(0.9).squeeze()
    test_excess_cdf = test_excess_pdf.cdf(bin_centers).squeeze()

    # ThermalNode.view_cdf(xlabel='Excess')
    # ThermalNode.axC.plot(bin_centers,test_excess_cdf,'--r')
    # ThermalNode.figC.show()

    assert np.math.isclose(reliability, test_reliability, rel_tol=1e-1)
    assert np.math.isclose(excess_limit, test_excess_limit, rel_tol=1e-1)
    assert np.allclose(cdf, test_excess_cdf, atol=1e-1)

if __name__ == "__main__":
    test_stochastic(stochastic_inputs)