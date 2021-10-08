import pytest
import numpy as np
import matplotlib.pyplot as plt

from dmLib.DOELib.Design import Design
from dmLib.uncertaintyLib.probFunction import guassianFunc

def test_multivariateGaussian():
    """testing the multivariate Gaussian function"""

    mu = np.array([0.5, 0.5]) # means vector
    Sigma = np.array(
        [[0.5,0.1],
        [0.1,0.5]]) # Covariance matrix

    lb = np.array([0.0, 0.0,])
    ub = np.array([1.0, 1.0,])
    s = Design(lb,ub,5,'fullfact').unscale()

    test_function = guassianFunc(mu,Sigma)

    test = test_function.multivariateGaussian(s)

    # Calculate probability density for a bivariate Gaussian function
    x1 = s[:,0]
    x2 = s[:,1]
    mu_1 = mu[0]
    mu_2 = mu[1]
    sigma_1 = np.sqrt(Sigma[0,0])
    sigma_2 = np.sqrt(Sigma[1,1])
    var_12 = Sigma[0,1]

    rho = var_12 / (sigma_1 * sigma_2)
    den = 2 * np.pi * sigma_1 * sigma_2 * np.sqrt(1-rho**2)
    N = 1 / den

    f1 = ((x1-mu_1)**2) / (sigma_1**2)
    f2 = (2*rho * (x1-mu_1) * (x2-mu_2)) / (sigma_1 * sigma_2)
    f3  = ((x2-mu_2)**2) / (sigma_2**2) 

    z = f1 - f2 + f3
    output = N * np.exp(-z / (2 * (1-rho**2)))

    assert np.allclose(test, output, rtol=1e-05)

def test_guassianVolume():
    """testing the multivariate Gaussian volume function"""

    mu = np.array([0.5, 0.5]) # means vector
    Sigma = np.array(
        [[0.075,0.0375],
        [0.0375,0.15]]) # Covariance matrix

    w, v = np.linalg.eig(Sigma) # eigen values and vector
    test_function = guassianFunc(mu,Sigma)
    r = 2

    # plot rotated ellipse
    t = np.linspace(0,2*np.pi,100)
    ellipsis = (np.sqrt(w[None,:]) * v * r) @ [np.sin(t), np.cos(t)] + np.tile(mu.reshape(2,1), (1,100))
    sigma_loc = ellipsis.T[0,:][None,:]
    p_sigma_loc = test_function.multivariateGaussian(sigma_loc)[0]

    # Compute volumes
    n_levels = 1000

    lb = np.array([-0.5, -1.0,])
    ub = np.array([1.5, 2.0,])
    s = Design(lb,ub,n_levels,type='fullfact').unscale()

    p = test_function.multivariateGaussian(s)
    V_test = (len(p[p>=p_sigma_loc]) / len(p)) * np.prod(ub - lb)
    V_output = test_function.compute_volume(r=r)

    # # Visualization
    # lb = np.array([-0.5, -1.0,])
    # ub = np.array([1.5, 2.0,])
    # s = Design(lb,ub,100,type='fullfact').unscale()
    # p = test_function.multivariateGaussian(s)

    # X1 = np.reshape(s[:,0],[100,100])
    # X2 = np.reshape(s[:,1],[100,100])
    # P = np.reshape(p,[100,100])

    # # Plot the result in 2D
    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot()

    # surf = ax.contourf(X1, X2, P, cmap=plt.cm.jet,)
    # ax.set_xlabel('x1')
    # ax.set_ylabel('x2')

    # # plot rotated ellipse
    # ax.plot(ellipsis[0,:], ellipsis[1,:], '-r', label='%i-$\sigma$' %r)
    # ax.legend()

    assert np.math.isclose(V_test, V_output, rel_tol=1e-1)