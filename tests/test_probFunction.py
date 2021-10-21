import pytest
import numpy as np
import matplotlib.pyplot as plt

from dmLib import Design, gaussianFunc, Distribution

# Input set 1
@pytest.fixture
def example_mean_std():

    mean = 10.0
    sd = 5.0

    return mean, sd

def test_compute_density():
    """
    testing the multivariate Gaussian function for 
    probability density computation
    """

    mu = np.array([0.5, 0.5]) # means vector
    Sigma = np.array(
        [[0.5,0.1],
        [0.1,0.5]]) # Covariance matrix

    lb = np.array([0.0, 0.0,])
    ub = np.array([1.0, 1.0,])
    s = Design(lb,ub,5,'fullfact').unscale()

    test_function = gaussianFunc(mu,Sigma)

    test = test_function.compute_density(s)

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

def test_compute_volume_2D():
    """
    testing the multivariate Gaussian 
    volume function on a 2D real space 
    for a skewed ellipse
    """

    mu = np.array([0.5, 0.5]) # means vector
    Sigma = np.array([
        [0.075,0.0],
        [0.0,0.15]
        ]) # Covariance matrix
    r = 2 # Mahalanobis distance

    # Compute volumes using compute_volume method
    test_function = gaussianFunc(mu,Sigma)
    V_output = test_function.compute_volume(r=r)

    # Compute volumes analytically
    s11 = np.sqrt(Sigma[0,0])
    s22 = np.sqrt(Sigma[1,1])
    V_test = np.pi * (r*s11*r*s22)

    assert np.math.isclose(V_test, V_output, rel_tol=1e-5)

    #################### Visualization ###################
    
    # # parametric curve for an ellipse
    # t = np.linspace(0,2*np.pi,100)
    # w, v = np.linalg.eig(Sigma) # eigen values and vector
    # ellipsis = (np.sqrt(w[None,:]) * v * r) @ [np.sin(t), np.cos(t)] + np.tile(mu.reshape(2,1), (1,100))

    # # probability density
    # n_levels = 100
    # lb = np.array([-0.5, -0.5,])
    # ub = np.array([1.5, 1.5,])
    # s = Design(lb,ub,n_levels,type='fullfact').unscale()
    # p = test_function.compute_density(s)

    # X1 = np.reshape(s[:,0],[n_levels,n_levels])
    # X2 = np.reshape(s[:,1],[n_levels,n_levels])
    # P = np.reshape(p,[n_levels,n_levels])

    # # Plot the result in 2D
    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot()

    # surf = ax.contourf(X1, X2, P, cmap=plt.cm.jet,)
    # ax.set_xlabel('x1')
    # ax.set_ylabel('x2')

    # # plot rotated ellipse
    # ax.plot(ellipsis[0,:], ellipsis[1,:], '-r', label='%i-$\sigma$' %r)
    # ax.legend()

def test_compute_volume_2D_cov():

    """
    testing the multivariate Gaussian 
    volume function on a 2D real space 
    for a skewed ellipse
    """

    mu = np.array([0.5, 0.5]) # means vector
    Sigma = np.array(
        [[0.075,0.0375],
        [0.0175,0.15]]) # Covariance matrix
    r = 2 # Mahalanobis distance

    # plot rotated ellipse
    test_function = gaussianFunc(mu,Sigma)
    V_output = test_function.compute_volume(r=r)

    # Compute volumes numerically
    
    # get probability density at r contour
    p_sigma_loc = test_function.compute_density_r(r=r)

    n_levels = 1000
    lb = np.array([-0.5, -1.0,])
    ub = np.array([1.5, 2.0,])
    s = Design(lb,ub,n_levels,type='fullfact').unscale()

    # Monte Carlo integration
    p = test_function.compute_density(s)
    V_test = (len(p[p>=p_sigma_loc]) / len(p)) * np.prod(ub - lb)

    assert np.math.isclose(V_test, V_output, rel_tol=1e-1)

    #################### Visualization ###################

    # # parametric curve for an ellipse
    # t = np.linspace(0,2*np.pi,100)
    # w, v = np.linalg.eig(Sigma) # eigen values and vector
    # ellipsis = (np.sqrt(w[None,:]) * v * r) @ np.array([np.cos(t), np.sin(t)]) + np.tile(mu.reshape(2,1), (1,100))

    # # probability density
    # n_levels = 100
    # lb = np.array([-0.5, -1.0,])
    # ub = np.array([1.5, 2.0,])
    # s = Design(lb,ub,n_levels,type='fullfact').unscale()
    # p = test_function.compute_density(s)

    # X1 = np.reshape(s[:,0],[n_levels,n_levels])
    # X2 = np.reshape(s[:,1],[n_levels,n_levels])
    # P = np.reshape(p,[n_levels,n_levels])

    # # Plot the result in 2D
    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot()

    # surf = ax.contourf(X1, X2, P, cmap=plt.cm.jet,)
    # ax.set_xlabel('x1')
    # ax.set_ylabel('x2')

    # # plot rotated ellipse
    # ax.plot(ellipsis[0,:], ellipsis[1,:], '-r', label='%i-$\sigma$' %r)
    # ax.plot([mu[0],v[0,0],], [mu[1],v[1,0],], '-r', label='%i-$\sigma$' %r)
    # ax.plot([mu[0],v[0,1],], [mu[1],v[1,1],], '-r', label='%i-$\sigma$' %r)
    # ax.legend()

def test_compute_density_r():
    """
    test the multivariate Gaussian function returns correct
    probability density at given Mahalanobis distance for 
    1D and 2D cases
    """
    
    mu = np.array([0.5,]) # means vector
    Sigma = np.array([
        [0.075,],
        ]) # Covariance matrix
    r = 2 # Mahalanobis distance

    # Compute density using compute_density_r method
    test_function = gaussianFunc(mu,Sigma)
    p_output = test_function.compute_density_r(r=r)

    # Compute density analytically
    sigma = np.sqrt(Sigma[0,0])
    m = mu[0]

    N  = np.sqrt(2 * np.pi) * sigma
    Z = np.exp( -(r**2) / 2 )

    p_test = Z / N

    assert np.math.isclose(p_test, p_output, rel_tol=1e-5)

    ##################### 2D example #####################

    mu = np.array([0.5, 0.5]) # means vector
    Sigma = np.array(
        [[0.075,0.0375],
        [0.0375,0.15]]) # Covariance matrix
    r = 2 # Mahalanobis distance

    # plot rotated ellipse
    test_function = gaussianFunc(mu,Sigma)
    p_output = test_function.compute_density_r(r=r)

    # Compute density numerically
    
    # parametric curve for an ellipse
    t = np.linspace(0,2*np.pi,100)
    w, v = np.linalg.eig(Sigma) # eigen values and vector
    ellipsis = (np.sqrt(w[None,:]) * v * r) @ [np.sin(t), np.cos(t)] + np.tile(mu.reshape(2,1), (1,100))
    
    # get probability density at r contour
    sigma_loc = ellipsis.T[0,:][None,:]
    p_test = test_function.compute_density(sigma_loc)[0]

    assert np.math.isclose(p_test, p_output, rel_tol=1e-5)

def test_gaussian_pdf_rvs(example_mean_std):
    """
    test the multivariate Gaussian pdf function 
    random generation capability
    """
    # # DEBUG:
    example_mean_std = [10.0,5.0]

    ######################################################

    mean,sd = example_mean_std

    # 1D example
    x = np.linspace(-100,100,100) # 2D grid
    dist = gaussianFunc(np.array([mean]),np.array([[sd**2,]]))
    p = dist.compute_density(x[:,None]) # get density values

    output_mean = dist(10000).mean(axis=1).squeeze() # should be close to 10.0
    output_sd = dist(10000).std(axis=1).squeeze() # should be close to 5.0

    if (output_mean.ndim == 0) & (output_sd.ndim == 0):
        output_mean = float(output_mean)
        output_sd = float(output_sd)
    else:
        raise ValueError("Number of dimensions in function output \
            and the test values is not the same")

    # # View distribution
    # import matplotlib.pyplot as plt
    # plt.hist(dist(10000).squeeze(), bins=100, density=True)
    # # plt.plot(np.linspace(-6,6),pdf)
    # plt.show()

    assert np.math.isclose(mean, output_mean, rel_tol=1e-1)
    assert np.math.isclose(sd, output_sd, rel_tol=1e-1)

    # 2D example
    mean,sd = example_mean_std

    mean = mean * np.ones(2)
    var = (sd**2) * np.eye(2)

    x = Design(-100*np.ones(2),100*np.ones(2),50,"fullfact").unscale() # 2D grid
    dist = gaussianFunc(mean,var)
    p = dist.compute_density(x) # get density values

    output_mean = dist(10000).mean(axis=1) # should be close to 10.0
    output_var = np.diag(dist(10000).var(axis=1)) # should be close to 5.0

    if (output_mean.ndim != 1) | (output_var.ndim != 2):
        raise ValueError("Number of dimensions in function output \
            and the test values is not the same")

    assert np.allclose(mean, output_mean, rtol=1e-1)
    assert np.allclose(var, output_var, rtol=1e-1)

    # # View distribution
    # import matplotlib.pyplot as plt
    # plt.scatter(*dist(1000))
    # plt.show()

def test_arbitrary_pdf_rvs(example_mean_std):
    """
    test the multivariate arbitrary pdf function 
    random generation capability
    """
    # # DEBUG:
    # example_mean_std = [10.0,5.0]

    ######################################################

    mean,sd = example_mean_std

    # 1D example
    x = np.linspace(-100,100,100) # 2D grid
    function = gaussianFunc(np.array([mean]),np.array([[sd**2,]]))
    p = function.compute_density(x[:,None]) # get density values
    dist = Distribution(p,lb=-100,ub=100)

    output_mean = dist(10000).mean(axis=1).squeeze() # should be close to 10.0
    output_sd = dist(10000).std(axis=1).squeeze() # should be close to 5.0

    if output_mean.ndim == 0 & output_sd.ndim == 0:
        output_mean = float(output_mean)
        output_sd = float(output_sd)
    else:
        raise ValueError("Number of dimensions in function output \
            and the test values is not the same")

    # # View distribution
    # import matplotlib.pyplot as plt
    # plt.hist(dist(10000).squeeze(), bins=100, density=True)
    # # plt.plot(np.linspace(-6,6),pdf)
    # plt.show()

    assert np.math.isclose(mean, output_mean, rel_tol=1e-1)
    assert np.math.isclose(sd, output_sd, rel_tol=1e-1)

    # 2D example
    mean,sd = example_mean_std

    mean = mean * np.ones(2)
    var = (sd**2) * np.eye(2)

    x = Design(-100*np.ones(2),100*np.ones(2),50,"fullfact").unscale() # 2D grid
    function = gaussianFunc(mean,var)
    p = function.compute_density(x) # get density values

    dist = Distribution(p.reshape((50,)*2),lb=-100,ub=100)
    output_mean = dist(10000).mean(axis=1) # should be close to 10.0
    output_var = np.diag(dist(10000).var(axis=1)) # should be close to 5.0

    if (output_mean.ndim != 1) | (output_var.ndim != 2):
        raise ValueError("Number of dimensions in function output \
            and the test values is not the same")

    if (len(output_mean) != 2) | (output_var.shape != (2,2)):
        raise ValueError("Number of dimensions in function output \
            and the test values is not the same")

    assert np.allclose(mean, output_mean, rtol=1e-1)
    assert np.allclose(var, output_var, rtol=1e-1)

    # # View distribution
    # import matplotlib.pyplot as plt
    # plt.scatter(*dist(1000))
    # plt.show()

    # 3D example
    mean,sd = example_mean_std

    mean = mean * np.ones(3)
    var = (sd**2) * np.eye(3)

    x = Design(-100*np.ones(3),100*np.ones(3),50,"fullfact").unscale() # 3D grid
    function = gaussianFunc(mean,var)
    p = function.compute_density(x) # get density values

    dist = Distribution(p.reshape((50,)*3),lb=-100,ub=100)
    output_mean = dist(10000).mean(axis=1) # should be close to 10.0
    output_var = np.diag(dist(10000).var(axis=1)) # should be close to 5.0

    if (output_mean.ndim != 1) | (output_var.ndim != 2):
        raise ValueError("Number of dimensions in function output \
            and the test values is not the same")

    if (len(output_mean) != 3) | (output_var.shape != (3,3)):
        raise ValueError("Number of dimensions in function output \
            and the test values is not the same")

    assert np.allclose(mean, output_mean, rtol=1e-1)
    assert np.allclose(var, output_var, rtol=1e-1)
    
if __name__ == "__main__":
    # debug a particular test
    inputs = [10.0,5.0]
    # test_gaussian_pdf_rvs(example_mean_std)
    test_arbitrary_pdf_rvs(inputs)