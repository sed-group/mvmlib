import numpy as np
from typing import Dict, Any, AnyStr, List, Type
import matplotlib.pyplot as plt
from .uncertaintyLib import Distribution
from .utilities import check_folder

"""Design margins library for computing buffer and excess"""
class MarginNode():

    def __init__(self,label='',cutoff=0.9,buffer_limit=0):
        """
        Contains description and implementation 
        of a Margin Node object which is the building block
        of a Margin Analysis Network (MAN)

        Parameters
        ----------
        cutoff : float, optional
            cutoff limit for calculating reliability,
            default = 0.9
        buffer_limit : float, optional
            lower bound for beginning of buffer zone,
            default = 0.0
        label : str, optional
            string to tag instance with, default = ''
        """

        self.label = label
        self.cutoff = 0.9
        self.buffer_limit = 0.0
        self._target = np.empty(0)
        self._response = np.empty(0)
        self._excess = np.empty(0)
        self._excess_dist = None

    @property
    def target(self):
        """
        Target vector getter

        Returns
        -------
        np.1darray
            vector of target observations
        """
        return self._target

    @target.setter
    def target(self,t):
        """
        Appends target observation t to target vector

        Parameters
        ----------
        t : float
            value to append to target vector
        """
        self._target = np.append(self._target,t)

    @property
    def response(self):
        """
        Response vector getter

        Returns
        -------
        np.1darray
            vector of response observations
        """
        return self._response

    @response.setter
    def response(self,r):
        """
        Appends response observation r to target vector

        Parameters
        ----------
        r : float
            value to append to response vector
        """
        self._response = np.append(self._response,r)

    @property
    def excess(self):
        """
        Excess vector getter

        Returns
        -------
        np.1darray
            vector of excess observations
        """
        return self._excess

    @excess.setter
    def excess(self,e):
        """
        Appends excess observation e to target vector

        Parameters
        ----------
        e : float
            value to append to response vector
        """
        self._excess = np.append(self._excess,e)

    @property
    def excess_dist(self) -> Distribution:
        """
        Excess Distribution object

        Returns
        -------
        dmLib.Distribution
            instance of dmLib.Distribution holding excess pdf
        """
        return self._excess_dist

    @excess_dist.setter
    def excess_dist(self,excess):
        """
        Creates excess Distribution object

        Parameters
        ----------
        excess : np.1darray
            Vector of excess values
        """
        self._excess_dist = Distribution(excess, lb=min(excess),ub=max(excess))

    def threshold(self, *args, **kwargs):
        """
        The function that will be used to calculate the target value
            - Can be a deterministic model
            - Can be a stochastic model (by calling a defined dmLib.Distribution instance)
        
        This method must be redefined by the user for every instance

        Example
        -------
        >>> # [in the plugin file]
        >>> from dmLib import designSolution
        >>> class myDesignSolution(designSolution):
        >>>     def threshold(self):
        >>>         # some threshold
        >>>         return t() # if t is a dmLib.Distribution
        """
        # default code for the default threshold
        return

    def behaviour(self, *args, **kwargs):
        """
        The function that will be used to calculate decided value
            - Can be a deterministic model
            - Can be a stochastic model (by calling a defined dmLib.Distribution instance)
        
        This method must be redefined by the user for every instance

        Example
        -------
        >>> # [in the plugin file]
        >>> from dmLib import designSolution
        >>> class myDesignSolution(designSolution):
        >>>     def behaviour(self,r,d):
        >>>         # some specific model-dependent behaviour
        >>>         return r*2+1 / d
        """
        # default code for the default behaviour
        return

    def reset(self):
        """
        Resets accumilated random observations in target, 
        response, and excess attributes
        """
        self._target = np.empty(0)
        self._response = np.empty(0)
        self._excess = np.empty(0)
        self._excess_dist = None

    def compute_cdf(self,bins=500):
        """
        Calculate the cumulative distribution function for the excess margin

        Parameters
        ----------
        bins : int, optional
            number of discrete bins used to 
            construct pdf and pdf curves, by default 500

        Returns
        -------
        bin_centers : np.1darray
            array of len(excess) - 1 containing the x-axis values of CDF
        cdf : np.1darray
            array of len(excess) - 1 containing the y-axis values of CDF
        excess_limit : float
            The value on the x-axis that corresponds to the cutoff probability
        reliability : float
            the probability of excess being >= target
        """
        def moving_average(x, w):
            """
            N-moving average over 1D array

            Parameters
            ----------
            x : np.1darray
                input array to average
            w : int
                number of elements to average

            Returns
            -------
            np.1darray
                avaeraged array
            """
            return np.convolve(x, np.ones(w), 'valid') / w

        excess_hist = np.histogram(self.excess, bins=bins,density=True)
        bin_width = np.mean(np.diff(excess_hist[1]))
        bin_centers = moving_average(excess_hist[1],2)
        cdf = np.cumsum(excess_hist[0] * bin_width)

        excess_limit = bin_centers[cdf >= self.cutoff][0]
        reliability = 1 - cdf[bin_centers >= self.buffer_limit][0]

        return bin_centers, cdf, excess_limit, reliability

    def view(self,xlabel='',savefile=None):
        """
        view 1D or 2D plot of probability distribution of excess
        """

        self.fig, self.ax = plt.subplots(figsize=(8, 3))
        self.ax.hist(self.excess, bins=100, density=True)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel('density')

        if savefile is not None:
            # Save figure to image
            check_folder('images/%s' %(self.label))
            self.fig.savefig('images/%s/%s.pdf' %(self.label,savefile), 
                format='pdf', dpi=200, bbox_inches='tight')

        plt.show()

    def view_cdf(self,xlabel='',savefile=None):
        """
        view 1D or 2D plot of cumulative distribution of excess
        """

        # calculate CDF
        bin_centers, cdf, excess_limit, reliability = self.compute_cdf()
        buffer_band = (bin_centers >= self.buffer_limit) & (cdf <= self.cutoff)
        excess_band = cdf >= self.cutoff

        self.figC, self.axC = plt.subplots(figsize=(8, 3))
        self.axC.plot(bin_centers, cdf, '-b')
        self.axC.vlines([self.buffer_limit,excess_limit],[0,0],[1-reliability,self.cutoff],linestyles='dashed')
        self.axC.fill_between(bin_centers[buffer_band], 0, cdf[buffer_band], facecolor='Green', alpha=0.4, label='Buffer')
        self.axC.fill_between(bin_centers[excess_band], 0, cdf[excess_band], facecolor='Red', alpha=0.4, label='Excess')
        
        tb = self.axC.text((excess_limit + self.buffer_limit) / 2 - 0.2, 0.1, 'Buffer', fontsize=14)
        te = self.axC.text((excess_limit + bin_centers[-1]) / 2 - 0.2, 0.1, 'Excess', fontsize=14)
        tb.set_bbox(dict(facecolor='white'))
        te.set_bbox(dict(facecolor='white'))

        self.axC.set_xlabel(xlabel)
        self.axC.set_ylabel('Cumulative density')

        if savefile is not None:
            # Save figure to image
            check_folder('images/%s' %(self.label))
            self.figC.savefig('images/%s/%s.pdf' %(self.label,savefile), 
                format='pdf', dpi=200, bbox_inches='tight')

        plt.show()

    def __call__(self,inputs,decisions):
        """
        Calculate behaviour and target for each input provided

        Parameters
        ----------
        inputs : np.ndarray
            Inputs to the margin node. They are mapped using the 
            behaviour() method to response. The dimensions are N * n_samples, 
            where N is the number of inputs to the node
        decisions : np.1darray
            The design parameters used to select the design at the margin node
            The length of this vector equals the number of design parameters
        """
        if inputs.ndim == 1:
            inputs = inputs[None,:] # reshape 1D arrays to 2D

        for i in range(inputs.shape[1]):
            
            r = self.behaviour(inputs[:,i],decisions)
            self.response = r # add to list of responses

            t = self.threshold(inputs[:,i],decisions)
            self.target = t # add to list of targets

            e = r - t
            self.excess = e # add to list of targets
