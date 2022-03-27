from typing import Union

import matplotlib.pyplot as plt
import numpy as np

from .utilities import check_folder
from .DOELib import Design

"""Uncertainty Library for computing different PDFs"""


def moving_average(i: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    N-moving average over 1D array

    Parameters
    ----------
    i : np.ndarray
        input array to average
    w : int
        number of elements to average

    Returns
    -------
    np.ndarray
        avaeraged array
    """
    return np.convolve(i, np.ones(w), 'valid') / w


def compute_cdf(values: np.ndarray, bins: int = 500, cutoff: float = None, buffer_limit: float = None):
    """
    Calculate the cumulative distribution function for the excess margin

    Parameters
    ----------
    values : np.ndarray
        Values to construct cumulative pdf from
    bins : int, optional
        number of discrete bins used to 
        construct pdf and pdf curves, by default 500
    cutoff : float, optional
        cutoff limit for calculating reliability, by default None
    buffer_limit : float, optional
        lower bound for beginning of buffer zone, by default None

    Returns
    -------
    values : np.ndarray
        vector of value samples to use for computing the cdf
    bin_centers : np.ndarray
        array of len(value) - 1 containing the x-axis values of CDF
    cdf : np.ndarray
        array of len(value) - 1 containing the y-axis values of CDF

    Raises
    ------
    AssertionError
        if only one of `cutoff` or `buffer_limit` is provided
    """

    value_hist = np.histogram(values, bins=bins, density=True)
    bin_width = np.mean(np.diff(value_hist[1]))
    bin_centers = moving_average(value_hist[1], 2)
    cdf = np.cumsum(value_hist[0] * bin_width)

    if all([cutoff is not None, buffer_limit is not None]):
        value_limit = bin_centers[cdf >= cutoff][0]
        reliability = 1 - cdf[bin_centers >= buffer_limit][0]
        return bin_centers, cdf, value_limit, reliability
    elif all([cutoff is None, buffer_limit is None]):
        return bin_centers, cdf
    else:
        raise AssertionError('You must provide both cutoff and buffer_limit, only one of them is provided')


class Distribution(object):
    def __init__(self, pdf: np.ndarray, lb=-1, ub=1, sort=True, interpolation=True, label=''):
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
        pdf : np.ndarray
            2d-array of shape n_samples * n_dims including 
            sample density values throughout the real or 
            discrete space
        lb : np.ndarray OR float OR int, optional
            The lower bound for the pdf support, default = -1
        ub : np.ndarray OR float OR int, optional
            The uppoer bound for the pdf support, default = 1
        sort : bool, optional
            if True sort pdf to avoid interpolation 
            errors when evaluating from cdf, by default True
        interpolation : bool, optional
            If true, treats input density values as 
            coming from a piecewise continuous distribution 
            If false, then a discrete distribution is assumed,
            by default True
        label : str, optional
            string to tag instance with   
        """

        self.ax = None
        self.fig = None
        self.shape = pdf.shape
        self.sort = sort
        self.interpolation = interpolation
        self.pdf = pdf
        self.lb = lb
        self.ub = ub
        self.label = label
        self.sample = None
        self._samples = np.empty((self.ndim, 0))

        # upper bound cannot be smaller than lower bound
        assert (self._ub >= self._lb).all()
        # Check that the interval is square
        assert ((self._ub - self._lb) == (self._ub[0] - self._lb[0])).all()

    @property
    def pdf(self):
        """
        Returns pdf values

        Returns
        -------
        np.ndarray OR np.ndarray
            array of pdf values
        """

        return self._pdf

    @pdf.setter
    def pdf(self, pdf):
        """
        Sets the sorted pdf and asserts it is positive

        Parameters
        ----------
        pdf: np.ndarray OR np.ndarray
            Array of pdf values. If ndim > 1 then 
            pdf must have a shape of size (n,n,...,n)
        """

        assert (np.all(pdf >= 0))

        # sort the PDF by magnitude
        if self.sort:
            self.sortindex = np.argsort(pdf.ravel(), axis=None)
            self._pdf = pdf.ravel()[self.sortindex]
        else:
            self._pdf = pdf.ravel()

    @property
    def cdf(self):
        """
        construct the cumulative distribution function

        Returns
        -------
        np.ndarray
            vector of cumilative distribution
        """

        return np.cumsum(self.pdf)

    @property
    def lb(self):
        """
        Returns the lower bound of the pdf supports

        Returns
        -------
        np.ndarray
            lower pdf supports
        """

        return self._lb

    @lb.setter
    def lb(self, lb):
        """
        Sets the lb parameter and asserts it is compatible with pdf

        Parameters
        ----------
        np.ndarray OR float OR int
            lower pdf support(s)
        """

        # convert to floats
        if isinstance(lb, (int, float)):
            self._lb = lb * np.ones(self.ndim)
        else:
            self._lb = lb  # reshape 1D arrays to 2D

        # number of dimensions in lower and upper bounds must be equal
        assert self._lb.shape == (self.ndim,)

    @property
    def ub(self):
        """
        Returns the upper bound of the pdf supports

        Returns
        -------
        np.ndarray
            upper pdf supports
        """

        return self._ub

    @ub.setter
    def ub(self, ub):
        """
        Sets the ub parameter and asserts it is compatible with pdf

        Parameters
        ----------
        np.ndarray OR float OR int
            upper pdf support(s)
        """

        if isinstance(ub, (int, float)):
            self._ub = ub * np.ones(self.ndim)
        else:
            self._ub = ub  # reshape 1D arrays to 2D

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

    @property
    def samples(self) -> np.ndarray:
        """
        Sample vector getter

        Returns
        -------
        np.ndarray
            vector of sample observations
        """
        return self._samples

    @samples.setter
    def samples(self, s: np.ndarray):
        """
        Appends target observation t to samples vector

        Parameters
        ----------
        s : np.ndarray
            value to append to samples vector
        """
        self._samples = np.append(self._samples, s, axis=1)

    def reset(self):
        """
        Resets the stored samples
        """
        self._samples = np.empty((self.ndim, 0))

    def transform(self, i):
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

        half_interval = np.tile(((self.ub - self.lb) / 2)[:, None], (1, i.shape[1]))
        half_mean = np.tile(((self.ub + self.lb) / 2)[:, None], (1, i.shape[1]))

        return (((i - self.shape[0] / 2) / (self.shape[0] / 2))
                * half_interval) + half_mean

    def view(self, xlabel='', savefile=None):
        """
        view 1D or 2D plot of distribution for visual checks

        Parameters
        ----------
        xlabel : str, optional
            variable names to be shown on plot axes, by default ''
        savefile : str, optional
            if provided saves an image of the figure in directory 
            /images/self.label/, by default None

        Raises
        ------
        ValueError
            if called when ndim is > 2
        """

        self.fig, self.ax = plt.subplots(figsize=(8, 3))

        if self.ndim == 1:
            # 1D example
            self.ax.hist(self.samples.squeeze(), bins=100, density=True)
            self.ax.set_xlabel(xlabel)
            self.ax.set_ylabel('density')

        elif self.ndim == 2:
            # View distribution
            self.ax.scatter(*self.samples)
            self.ax.set_title(xlabel)

        else:
            raise ValueError("only applicable for 1D and 2D probability density functions")

        if savefile is not None:
            # Save figure to image
            check_folder('images/%s' % self.label)
            self.fig.savefig('images/%s/%s.pdf' % (self.label, savefile),
                             format='pdf', dpi=200, bbox_inches='tight')

        plt.show()

    def random(self, n=1):
        """
        draw random samples from PDF

        Parameters
        ----------
        n : int, optional
            Number of random samples to draw
            default is one sample

        Returns
        -------
        np.ndarray
            A 2D array of shape ndim * N, where 
            N is the number of requested samples
        """

        # pick numbers which are uniformly random over the cumulative distribution function
        choice = np.random.uniform(high=self.sum, size=n)
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

        self.sample = self.transform(index)  # store the sample inside instance
        self.samples = self.sample  # append sample to the samples inside instance

        if self.ndim == 1:
            self.sample = self.sample.squeeze(0)

        return self.sample  # return the requested number of samples

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
            copy of Distribution instance
        """
        id_self = id(self) # memoization avoids unnecessary recursion
        _copy = memo.get(id_self)
        if _copy is None:

            _copy = type(self)(self.pdf, self.lb, self.ub, self.sort, self.interpolation, self.label)
            _copy.label = self.label+'_copy_'+str(id(_copy))
            _copy._samples = self._samples
            memo[id_self] = _copy 
        return _copy


class GaussianFunc(Distribution):

    def __init__(self, mu: Union[float, int, np.ndarray], sigma: Union[float, int, np.ndarray], label=''):
        """
        Contains description and implementation of the multivariate 
        Gaussian PDF

        Parameters
        ----------
        mu : np.ndarray
            1d array of length n_dims containing means
        sigma : np.ndarray
            1d array of length n_dims containing standard deviations
            OR
            2d array of length n_dims * n_dims containing standard deviations
            and covariances
        label : str, optional
            string to tag instance with        
        """

        if type(mu) == float or type(mu) == int:
            mu = np.array([float(mu), ])
        if type(sigma) == float or type(mu) == int:
            sigma = np.array([[float(sigma), ], ])

        self.mu = mu
        self.Sigma = sigma
        lb = self.mu - 3 * np.sqrt(np.max(self.eigvals))
        ub = self.mu + 3 * np.sqrt(np.max(self.eigvals))

        x = Design(lb, ub, 50, "fullfact").unscale()  # 2D grid
        p = self.compute_density(x)  # get density values
        pdf = p.reshape((50,) * self.ndim)

        super().__init__(pdf, lb=lb, ub=ub, label=label)  # Initialize a distribution object for calling random samples

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
        np.ndarray
            eigen values
        """

        return np.linalg.eigvals(self.Sigma)

    def compute_density(self, samples):
        """
        Return the multivariate Gaussian probability density 
        distribution on array samples.

        Parameters
        ----------
        samples : np.ndarray
            array of shape n_samples * n_dims at which PDF will be evaluated

        Returns
        -------
        z : np.ndarray
            array of shape n_samples of PDF values
        """

        n_samples = samples.shape[0]

        # pos is an array constructed by packing the meshed arrays of variables
        # x_1, x_2, x_3, ..., x_k into its _last_ dimension.
        pos = np.empty((n_samples, 1) + (self.ndim,))

        for i in range(self.ndim):
            x_norm = np.reshape(samples[:, i], (n_samples, 1))
            # Pack X1, X2 ... Xk into a single 3-dimensional array
            pos[:, :, i] = x_norm

        sigma_inv = np.linalg.inv(self.Sigma)
        sigma_det = np.linalg.det(self.Sigma)

        n = np.sqrt((2 * np.pi) ** self.ndim * sigma_det)

        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) - the Mahalanobis distance d^2 - in a vectorized
        # way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', samples - self.mu, sigma_inv, samples - self.mu)

        z = np.exp(-fac / 2)

        return z / n

    def compute_volume(self, r=3):
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
            v_d = (np.pi ** (self.ndim / 2)) / np.math.factorial(self.ndim / 2)  # if self.ndim is even
        else:
            v_d = (2 ** self.ndim) * (np.pi ** ((self.ndim - 1) / 2)) / \
                  (np.math.factorial((self.ndim - 1) / 2) / np.math.factorial(self.ndim))  # if self.ndim is odd

        return v_d * np.power(np.linalg.det(self.Sigma), 0.5) * (r ** self.ndim)

    def compute_density_r(self, r=3):
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
        sigma_det = np.linalg.det(self.Sigma)
        n = np.sqrt((2 * np.pi) ** self.ndim * sigma_det)

        return np.exp(-r ** 2 / 2) / n

    def view(self, xlabel='', savefile=None):
        """
        view 1D or 2D plot of distribution for visual checks

        Parameters
        ----------
        xlabel : str, optional
            variable names to be shown on plot axes, by default ''
        savefile : str, optional
            if provided saves an image of the figure in directory 
            /images/self.label/, by default None

        Raises
        ------
        ValueError
            if called when ndim is > 2
        """
        super().view(xlabel=xlabel, savefile=savefile)  # call parent distribution class

        if self.ndim == 1:  # add trace of normal distribution to plot

            x = np.linspace(self.lb.squeeze(), self.ub.squeeze(), 100)  # 1D grid
            p = self.compute_density(x[:, None])  # get density values
            self.ax.plot(x, p)
            plt.draw()
            plt.pause(0.0001)

        if savefile is not None:
            # Save figure to image
            check_folder('images/%s' % self.label)
            self.fig.savefig('images/%s/%s.pdf' % (self.label, savefile),
                             format='pdf', dpi=200, bbox_inches='tight')

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
            copy of Distribution instance
        """
        id_self = id(self) # memoization avoids unnecessary recursion
        _copy = memo.get(id_self)
        if _copy is None:
            
            _copy = type(self)(self.mu, self.sigma, self.label)
            _copy.label = self.label+'_copy_'+str(id(_copy))
            _copy._samples = self._samples
            memo[id_self] = _copy 
        return _copy

class UniformFunc(Distribution):

    def __init__(self, center: Union[float, int, np.ndarray], interval: Union[float, int, np.ndarray], label=''):
        """
        Contains description and implementation of the multivariate 
        Gaussian PDF

        Parameters
        ----------
        center : np.ndarray
            1d array of length n_dims containing uniform distribution centers
        interval : np.ndarray
            1d array of length n_dims containing spread along each dimension
        label : str, optional
            string to tag instance with        
        """

        if type(center) == float or type(center) == int:
            center = np.array([float(center), ])
        if type(interval) == float or type(center) == int:
            interval = np.array([[float(interval), ], ])

        self.center = center
        self.Range = interval
        lb = self.center - 1.5 * np.linalg.norm(interval)
        ub = self.center + 1.5 * np.linalg.norm(interval)

        x = Design(lb, ub, 50, "fullfact").unscale()  # 2D grid
        p = self.compute_density(x)  # get density values
        pdf = p.reshape((50,) * self.ndim)

        super().__init__(pdf, lb=lb, ub=ub, label=label)  # Initialize a distribution object for calling random samples

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

        return len(self.center)

    def compute_density(self, samples):
        """
        Return the multivariate Gaussian probability density 
        distribution on array samples.

        Parameters
        ----------
        samples : np.ndarray
            array of shape n_samples * n_dims at which PDF will be evaluated

        Returns
        -------
        Z : np.ndarray
            array of shape n_samples of PDF values
        """

        n_samples = samples.shape[0]
        density = np.zeros(n_samples)
        density[np.all(samples <= (self.center + self.Range), axis=1) & np.all(samples >= (self.center - self.Range),
                                                                               axis=1)] = 1 / self.compute_volume()

        return density

    def compute_volume(self) -> float:
        """
        The volume of the rectangle given by Range

        Returns
        -------
        V : float
            volume of hyperrectangle

        """

        return np.prod(2 * self.Range)

    def view(self, xlabel='', savefile=None):
        """
        view 1D or 2D plot of distribution for visual checks

        Parameters
        ----------
        xlabel : str, optional
            variable names to be shown on plot axes, by default ''
        savefile : str, optional
            if provided saves an image of the figure in directory 
            /images/self.label/, by default None

        Raises
        ------
        ValueError
            if called when ndim is > 2
        """
        super().view(xlabel=xlabel, savefile=savefile)  # call parent distribution class

        if self.ndim == 1:  # add trace of normal distribution to plot

            x = np.linspace(self.lb.squeeze(), self.ub.squeeze(), 100)  # 1D grid
            p = self.compute_density(x[:, None])  # get density values
            self.ax.plot(x, p)
            plt.draw()
            plt.pause(0.0001)

        if savefile is not None:
            # Save figure to image
            check_folder('images/%s' % self.label)
            self.fig.savefig('images/%s/%s.pdf' % (self.label, savefile),
                             format='pdf', dpi=200, bbox_inches='tight')

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
            copy of Distribution instance
        """
        id_self = id(self) # memoization avoids unnecessary recursion
        _copy = memo.get(id_self)
        if _copy is None:
            
            _copy = type(self)(self.center, self.Range, self.label)
            _copy.label = self.label+'_copy_'+str(id(_copy))
            _copy._samples = self._samples
            memo[id_self] = _copy 
        return _copy

class VisualizeDist:
    def __init__(self, values: np.ndarray, cutoff: float = None, buffer_limit: float = None):
        """
        Contains PDF and CDF visualization tools

        Parameters
        ----------
        values : np.ndarray
            vector of sample observations whose PDF is to be visualized using a histogram
        cutoff : float, optional
            cutoff limit for calculating reliability, by default None
        buffer_limit : float, optional
            lower bound for beginning of buffer zone, by default None
        """

        self.values = values
        self.cutoff = cutoff
        self.buffer_limit = buffer_limit

    def view(self, xlabel: str = '', folder: str = '', file: str = None, img_format: str = 'pdf'):
        """
        view 1D or 2D plot of probability distribution of value

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

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(self.values, bins=100, density=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('density')

        if file is not None:
            # Save figure to image
            check_folder('images/%s' % folder)
            fig.savefig('images/%s/%s.%s' % (folder, file, img_format),
                        format=img_format, dpi=200, bbox_inches='tight')

        plt.show()

    def view_cdf(self, xlabel: str = '', folder: str = '', file: str = None, img_format: str = 'pdf'):
        """
        view 1D or 2D plot of probability distribution of value

        Parameters
        ----------
        xlabel : str, optional
            the xlabel to display on the plot, by default ''
        folder : str, optional
            folder in which to store image, by default ''
        file : str, optional
            name of image file, if not provide then an image is not saved, by default None
        img_format : str, optional
            format of the image to be stored, by default 'pdf'
        """

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Cumulative density')

        # calculate CDF
        if all([self.cutoff is not None, self.buffer_limit is not None]):
            bin_centers, cdf, value_limit, reliability = compute_cdf(self.values, cutoff=self.cutoff,
                                                                     buffer_limit=self.buffer_limit)
            buffer_band = (bin_centers >= self.buffer_limit) & (cdf <= self.cutoff)
            value_band = cdf >= self.cutoff

            ax.vlines([self.buffer_limit, value_limit], [0, 0], [1 - reliability, self.cutoff], linestyles='dashed')
            ax.fill_between(bin_centers[buffer_band], 0, cdf[buffer_band], facecolor='Green', alpha=0.4, label='Buffer')
            ax.fill_between(bin_centers[value_band], 0, cdf[value_band], facecolor='Red', alpha=0.4, label='Excess')

            tb = ax.text((value_limit + self.buffer_limit) / 2 - 0.2, 0.1, 'Buffer', fontsize=14)
            te = ax.text((value_limit + bin_centers[-1]) / 2 - 0.2, 0.1, 'Excess', fontsize=14)
            tb.set_bbox(dict(facecolor='white'))
            te.set_bbox(dict(facecolor='white'))

        else:
            bin_centers, cdf = compute_cdf(self.values)

        ax.plot(bin_centers, cdf, '-b')

        if file is not None:
            # Save figure to image
            check_folder('images/%s' % folder)
            fig.savefig('images/%s/%s.%s' % (folder, file, img_format),
                        format=img_format, dpi=200, bbox_inches='tight')

        plt.show()


if __name__ == "__main__":
    from mvm import Design, GaussianFunc

    mean = 10.0
    sd = 5.0

    # 1D example
    x_test = np.linspace(-100, 100, 100)  # 2D grid
    function = GaussianFunc(np.array([mean]), np.array([[sd ** 2, ]]))
    p_test = function.compute_density(x_test[:, None])  # get density values

    dist = Distribution(p_test)
    print(dist.random(10000).mean(axis=1))  # should be close to 10.0
    print(dist.random(10000).std(axis=1))  # should be close to 5.0

    # View distribution
    plt.hist(dist.random(10000).squeeze(), bins=100, density=True)
    plt.show()

    # 2D example
    x_test = Design(np.array([-100, -100]), np.array([100, 100]), 512, "fullfact").unscale()  # 2D grid
    function = GaussianFunc(np.array([mean, mean]), np.array([[sd ** 2, 0], [0, sd ** 2]]))
    p_test = function.compute_density(x_test)  # get density values

    dist = Distribution(p_test.reshape((512, 512)))
    print(dist.random(1000000).mean(axis=1))  # should be close to 10.0
    print(dist.random(1000000).std(axis=1))  # should be close to 5.0

    # View distribution
    import matplotlib.pyplot as plt

    plt.scatter(*dist.random(1000))
    plt.show()
