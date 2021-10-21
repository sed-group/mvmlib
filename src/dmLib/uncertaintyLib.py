import numpy as np
import  scipy.stats as st
import matplotlib.pyplot as plt
from .DOELib import Design
"""Uncertainty Library for computing different PDFs"""
class Distribution(object):
    def __init__(self, pdf, lb = -1, ub = 1, sort = True, interpolation = True):
        """
        Draws samples from a one dimensional probability distribution,
        by means of inversion of a discrete inversion of a cumulative density function,
        the PDF can be sorted first to prevent numerical error in the cumulative sum
        this is set as default; for big density functions with high contrast,
        it is absolutely necessary, and for small density functions,
        the overhead is minimal,
        a call to this distribution object returns indices into density array,
        borrowed from: https://stackoverflow.com/a/21101584

        Parameters
        ----------
        lb : np.1darray OR float OR int, optional
            The lower bound for the pdf support, default = -1
        ub : np.1darray OR float OR int, optional
            The uppoer bound for the pdf support, default = 1
        pdf : np.ndarray
            2d-array of shape n_samples * n_dims including 
            sample density values throughout the real or 
            discrete space
        sort : bool, optional
            if True sort pdf to avoid interpolation 
            errors when evaluating from cdf, by default True
        interpolation : bool, optional
            If true, treats input density values as 
            coming from a piecewise continuous distribution 
            If false, then a discrete distribution is assumed,
            by default True
        """

        self.shape          = pdf.shape
        self.pdf            = pdf.ravel()
        self.sort           = sort
        self.interpolation  = interpolation
        self.lb = lb
        self.ub = ub

        # upper bound cannot be smaller than lower bound
        assert(self._ub > self._lb).all()
        # Check that the interval is square
        assert ((self._ub - self._lb) == (self._ub[0] - self._lb[0])).all()

        # a PDF can not be negative
        assert(np.all(pdf>=0))

        # sort the PDF by magnitude
        if self.sort:
            self.sortindex = np.argsort(self.pdf, axis=None)
            self.pdf = self.pdf[self.sortindex]
        # construct the cumulative distribution function
        self.cdf = np.cumsum(self.pdf)

    @property
    def lb(self):
        """
        Returns the lower bound of the pdf supports

        Returns
        -------
        np.1darray
            lower pdf supports
        """

        return self._lb

    @lb.setter
    def lb(self,lb):
        """
        Sets the lb parameter and asserts it is compatible with pdf

        Parameters
        ----------
        np.1darray OR float OR int
            lower pdf support(s)
        """

        # convert to floats
        if isinstance(lb, (int,float)): self._lb = lb * np.ones(self.ndim) 
        else: self._lb = lb # reshape 1D arrays to 2D

        # number of dimensions in lower and upper bounds must be equal
        assert self._lb.shape == (self.ndim,)

    @property
    def ub(self):
        """
        Returns the upper bound of the pdf supports

        Returns
        -------
        np.1darray
            upper pdf supports
        """

        return self._ub

    @ub.setter
    def ub(self,ub):
        """
        Sets the ub parameter and asserts it is compatible with pdf

        Parameters
        ----------
        np.1darray OR float OR int
            upper pdf support(s)
        """

        if isinstance(ub, (int,float)): self._ub = ub * np.ones(self.ndim) 
        else: self._ub = ub # reshape 1D arrays to 2D

        # number of dimensions in lower and upper bounds must be equal
        assert self._ub.shape == (self.ndim,)

    @property
    def ndim(self):
        """
        Returns the number of dimensions 
        from the input data

        Returns
        -------
        int
            number of dimensions
        """

        return len(self.shape)

    @property
    def sum(self):
        """
        cached sum of all PDF values; 
        the PDF need not sum to one, 
        and is implicitly normalized

        Returns
        -------
        float
            sum of all PDF values
        """

        return self.cdf[-1]

    def transform(self,i):
        """
        Transform discrete integer choices when sampling
        to their continues real valued random variable samples

        Parameters
        ----------
        np.ndarray
            Array of indices of shape ndim * N, 
            where N is the number of samples
        
        Returns
        -------
        np.ndarray
            Array of transformed indices of same shape as input i
        """

        half_interval = np.tile(((self.ub - self.lb)/2)[:,None],(1,i.shape[1]))
        half_mean = np.tile(((self.ub + self.lb) / 2)[:,None],(1,i.shape[1]))

        return ((((i - self.shape[0]/2)) / (self.shape[0]/2)) \
            * half_interval) + (half_mean)

    def view(self):
        """
        view 1D or 2D plot of distribution for visual checks
        """

        self.fig, self.ax = plt.subplots(figsize=(8, 3))

        if self.ndim == 1:
            # 1D example
            self.ax.hist(self.__call__(10000).squeeze(), bins=100, density=True)
            plt.show()

        elif self.ndim == 2:
            # View distribution
            self.ax.scatter(*self.__call__(1000))
            plt.show()

        else:
            raise ValueError("only applicable for 1D and 2D probability density functions") 

    def __call__(self, N=1):
        """
        draw random samples from PDF

        Parameters
        ----------
        N : int, optional
            Number of random samples to draw
            default is one sample

        Returns
        -------
        np.ndarray
            A 2D array of shape ndim * N, where 
            N is the number of requested samples
        """

        # pick numbers which are uniformly random over the cumulative distribution function
        choice = np.random.uniform(high = self.sum, size = N)
        # find the indices corresponding to this point on the CDF
        index = np.searchsorted(self.cdf, choice)

        # if necessary, map the indices back to their original ordering
        if self.sort:
            index = self.sortindex[index]
        # map back to multi-dimensional indexing
        index = np.unravel_index(index, self.shape)
        index = np.vstack(index)
        # is this a discrete or piecewise continuous distribution?
        if self.interpolation:
            index = index + np.random.uniform(size=index.shape)
        return self.transform(index)

class gaussianFunc(Distribution):

    def __init__(self,mu,Sigma,label=''):
        """
        Contains description and implementation of the multivariate 
        Gaussian PDF

        Parameters
        ----------
        mu : np.1darray
            1d array of length n_dims containing means
        Sigma : np.ndarray
            1d array of length n_dims containing standard deviations
            OR
            2d array of length n_dims * n_dims containing standard deviations
            and covariances
        label : str, optional
            string to tag instance with        
        """

        self.mu = mu
        self.Sigma = Sigma
        self.label = label
        lb = self.mu - 3 * np.sqrt(np.max(self.eigvals))
        ub = self.mu + 3 * np.sqrt(np.max(self.eigvals))

        x = Design(lb,ub,50,"fullfact").unscale() # 2D grid
        p = self.compute_density(x) # get density values
        pdf = p.reshape((50,)*self.ndim)
        
        super().__init__(pdf,lb=lb,ub=ub) # Initialize a distribution object for calling random samples

    @property
    def ndim(self):
        """
        Returns the number of dimensions 
        from the input data

        Returns
        -------
        int
            number of dimensions
        """

        return len(self.mu)

    @property
    def eigvals(self):
        """
        Returns the eigen values of the covariance matrix

        Returns
        -------
        np.1darray
            eigen values
        """

        return np.linalg.eigvals(self.Sigma)

    def compute_density(self,samples):
        """
        Return the multivariate Gaussian probability density 
        distribution on array samples.

        Parameters
        ----------
        samples : np.ndarray
            array of shape n_samples * n_dims at which PDF will be evaluated

        Returns
        -------
        Z : np.1darray
            array of shape n_samples * n_dims of PDF values
        """

        n_samples = samples.shape[0]

        # pos is an array constructed by packing the meshed arrays of variables
        # x_1, x_2, x_3, ..., x_k into its _last_ dimension.
        pos = np.empty((n_samples,1) + (self.ndim,))
            
        for i in range(self.ndim):
            X_norm = np.reshape(samples[:,i],(n_samples,1))
            # Pack X1, X2 ... Xk into a single 3-dimensional array
            pos[:, :, i] = X_norm

        Sigma_inv = np.linalg.inv(self.Sigma)
        Sigma_det = np.linalg.det(self.Sigma)

        N = np.sqrt((2*np.pi)**self.ndim * Sigma_det)
        
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) - the Mahalanobis distance d^2 - in a vectorized
        # way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', samples-self.mu, Sigma_inv, samples-self.mu)

        Z = np.exp(-fac / 2)

        return Z / N

    def compute_volume(self,r=3):
        """
        The volume of the ellipsoid (x-mu)T.Sigma-1.(x-mu) = r
        This is the output of this method.

        Parameters
        ----------
        r : float
            corresponds to Mahalanobis distance r for hyperellipsoids
            r = 1 ---> 1 sigma
            r = 2 ---> 2 sigma
            r = 3 ---> 3 sigma

        Returns
        -------
        V : float
            volume of hyperellipsoid for Mahalanobis distance r

        """
        
        if (self.ndim % 2) == 0:
            V_d = (np.pi**(self.ndim/2)) / np.math.factorial(self.ndim/2) # if self.ndim is even
        else:
            V_d = (2**(self.ndim)) * (np.pi**((self.ndim - 1)/2)) / \
                (np.math.factorial((self.ndim - 1)/2) / np.math.factorial(self.ndim)) # if self.ndim is odd

        return V_d * np.power(np.linalg.det(self.Sigma), 0.5) * (r**self.ndim)

    def compute_density_r(self,r=3):
        """
        Returns the value of probability density at given Mahalanobis distance r

        Parameters
        ----------
        r : float
            corresponds to Mahalanobis distance r for hyperellipsoids
            r = 1 ---> 1 sigma
            r = 2 ---> 2 sigma
            r = 3 ---> 3 sigma

        Returns
        -------
        p : float
            probability density at Mahalanobis distance r

        """
        Sigma_det = np.linalg.det(self.Sigma)
        N = np.sqrt((2*np.pi)**self.ndim * Sigma_det)

        return np.exp(-r**2 / 2)/N

    def view(self):
        """
        view 1D or 2D plot of distribution for visual checks
        """

        super().view() # call parent distribution class

        if self.ndim == 1: # add trace of normal distribution to plot
        
            x = np.linspace(self.lb.squeeze(), self.ub.squeeze(),100) # 1D grid
            p = self.compute_density(x[:,None]) # get density values
            self.ax.plot(x,p)
            plt.draw()
            plt.pause(0.0001)

if __name__ == "__main__":

    from dmLib import Design, gaussianFunc

    mean = 10.0
    sd = 5.0

    # 1D example
    x = np.linspace(-100,100,100) # 2D grid
    function = gaussianFunc(np.array([mean]),np.array([[sd**2,]]))
    p = function.compute_density(x[:,None]) # get density values

    dist = Distribution(p)
    print(dist(10000).mean(axis=1)) # should be close to 10.0
    print(dist(10000).std(axis=1)) # should be close to 5.0

    # View distribution
    plt.hist(dist(10000).squeeze(), bins=100, density=True)
    plt.show()

    # 2D example
    x = Design(np.array([-100,-100]),np.array([100,100]),512,"fullfact").unscale() # 2D grid
    function = gaussianFunc(np.array([mean,mean]),np.array([[sd**2,0],[0,sd**2]]))
    p = function.compute_density(x) # get density values

    dist = Distribution(p.reshape((512,512)))
    print(dist(1000000).mean(axis=1)) # should be close to 10.0
    print(dist(1000000).std(axis=1)) # should be close to 5.0

    # View distribution
    import matplotlib.pyplot as plt
    plt.scatter(*dist(1000))
    plt.show()