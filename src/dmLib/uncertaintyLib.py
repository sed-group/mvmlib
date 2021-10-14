import numpy as np

"""Uncertainty Library for computing different PDFs"""

class gaussianFunc():

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
        self.n_dims = len(self.mu)

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
        pos = np.empty((n_samples,1) + (self.n_dims,))
            
        for i in range(self.n_dims):
            X_norm = np.reshape(samples[:,i],(n_samples,1))
            # Pack X1, X2 ... Xk into a single 3-dimensional array
            pos[:, :, i] = X_norm

        Sigma_inv = np.linalg.inv(self.Sigma)
        Sigma_det = np.linalg.det(self.Sigma)

        N = np.sqrt((2*np.pi)**self.n_dims * Sigma_det)
        
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
        
        if (self.n_dims % 2) == 0:
            V_d = (np.pi**(self.n_dims/2)) / np.math.factorial(self.n_dims/2) # if self.n_dims is even
        else:
            V_d = (2**(self.n_dims)) * (np.pi**((self.n_dims - 1)/2)) / \
                (np.math.factorial((self.n_dims - 1)/2) / np.math.factorial(self.n_dims)) # if self.n_dims is odd

        return V_d * np.power(np.linalg.det(self.Sigma), 0.5) * (r**self.n_dims)

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
        N = np.sqrt((2*np.pi)**self.n_dims * Sigma_det)

        return np.exp(-r**2 / 2)/N

class probFunction(gaussianFunc):

    def __init__(self,mean,spread,type,label=''):
        """
        base class for managing different probability density functions

        Parameters
        ----------
        mean : np.1darray
            1d array of length n
            n=number of dimensions
        spread : np.ndarray
            1d array of length n
            n=number of dimensions

            OR

            2d array of length n * n containing standard deviations
            and covariances if "type" is "guassian"
        
        type : str
            Allowable values are "guassian" or "g", "uniform" or "u", 
        label : str, optional
            string to tag instance with
        """
        self.type = type
        self.n_dims = len(mean)

        if self.type == 'guassian':
            gaussianFunc().__init__(mean,spread,label)
            

    def getPDFValues(self,samples):
        """
        Return the value of the PDF at sampling sites.

        Parameters
        ----------
        samples : np.ndarray
            array of shape n_samples * n_dims at which PDF will be evaluated
        
        Returns
        -------
        Z : np.ndarray
            array of shape n_samples * n_dims of PDF values
        """

        if self.type == 'guassian':
            Z = gaussianFunc().multivariate_gaussian(self,samples)

        return Z

