# -*- coding: utf-8 -*-
"""
Neighborhood algorithm direct-search optimization
"""

from copy import copy, deepcopy
from random import uniform
import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot as plt
from .reference import rosenbrock


class Searcher():
    
    def __init__(self, objective, limits, num_samp, num_resamp, names=[], maximize=False, verbose=True):
        """
        Neighborhood algorithm direct-search optimization
        
        Arguments:
            objective: callable accepting a 1D numpy array of parameters, and
                returning a scalar misfit value
            limits: list of tuples defining range for each objective
                function parameter, as [(min_val, max_val), ...]
            num_samp: int, number of random samples taken at each iteration.
            num_resamp: int, number of best Voronoi polygons sampled at
                each iteration.
            names: list of strings, names for objective function parameters,
                used for plotting, and totally optional
            maximize: boolean, set True to maximize the objective function,
                or false to minimize it.
            verbose: set True to print verbose progress messages
        
        References:
            Sambridge, M. (1999). Geophysical inversion with a neighbourhood
            algorithm - I. Searching a parameter space. Geophysical Journal
            International, 138(2), 479–494.
        """
        # store and validate input args
        self._objective = objective
        self._limits = deepcopy(limits)
        self._num_samp = num_samp
        self._num_resamp = num_resamp
        if names:
            self._names = copy(names)
        else: 
            self._names = ['x{}'.format(ii) for ii in range(len(limits))]
        self._maximize = maximize
        self._verbose = verbose
        self._validate_args()

        # init constants and working vars
        self._num_dim = len(limits)
        self._param_min = np.array([x[0] for x in limits])
        self._param_max = np.array([x[1] for x in limits])
        self._param_rng = np.array([x[1]-x[0] for x in limits])
        self._sample = []
        self._queue = [] 
        self._iter = 0
        
    @property
    def sample(self):
        return deepcopy(self._sample)

    @property
    def sample_dataframe(self):
        samps = self.sample
        for samp in samps:
            for name, val in zip(self._names, samp['param']):
                samp[name] = val
            del samp['param']
        return pd.DataFrame(samps)

    def update(self, num_iter=10):
        """
        Execute search algorithm for specified number of iterations
        
        Arguments:
            num_iter: int, number of iterations to run
        """
        
        for ii in range(num_iter):
            
            # generate new sample (populates queue)
            if not self._sample:
                self._random_sample()
            else:
                self._neighborhood_sample()
                        
            # execute forward model for all samples in queue
            while self._queue:
                param = self._queue.pop()
                result = self._objective(param)
                self._sample.append({
                    'param': param,
                    'result': result,
                    'iter': self._iter
                    })
             
            # prepare for next iteration
            self._sample.sort(key=lambda x: x['result'], reverse=self._maximize)
            self._iter += 1
            if self._verbose:
                print(self)
    
    def plot(self):
        """Pair-plots of sample distribution"""
        df = self.sample_dataframe

        # get names of variables to plot
        var_names = set(df.columns)
        var_names.discard('iter')
        var_names.discard('result')
        var_names = list(var_names)
        num_vars = len(var_names)

        # get axis limits for each variable
        var_limits = {}
        for name in var_names:
            idx = self._names.index(name)
            var_limits[name] = self._limits[idx]

        # main loop, populate grid and titles
        fig, axs = plt.subplots(nrows=num_vars, ncols=num_vars)
        fig.subplots_adjust(hspace=0.2, wspace=0.2)
        for ii in range(num_vars):
            for jj in range(num_vars):
                ax = axs[ii][jj]
                if ii == jj: 
                    # diagonal, plot 1D KDE
                    series = df[var_names[ii]]
                    series.plot.kde(ax=ax, xlim=var_limits[var_names[ii]])
                    axs[ii][jj].set_yticklabels([], fontdict=None, minor=False)
                elif ii < jj:
                    # upper triangle, plot scatter colored by result value
                    xx = df[var_names[jj]]
                    yy = df[var_names[ii]]
                    zz = df['result']
                    ax.scatter(xx, yy, c=zz)
                    for tick in ax.get_yticklabels():
                            tick.set_rotation(45)
                elif ii > jj:
                    # lower triangle, plot 2D KDE
                    xx = df[var_names[jj]]
                    yy = df[var_names[ii]]
                    seaborn.kdeplot(xx, yy, ax=ax, cmap="Blues", shade=True, shade_lowest=True)
                    for tick in ax.get_yticklabels():
                            tick.set_rotation(45)
                    if ii != num_vars - 1:
                        axs[ii][jj].set_xlabel('')
                    if jj != 0:
                        axs[ii][jj].set_ylabel('')
        ttl = [
            '2D PDFs (lower), 1D PDFs (diag), 2D Scatter (upper)',
            'Best = {:.4f}'.format(df['result'][0]),
             ]
        fig.suptitle('\n'.join(ttl))

        fig.subplots_adjust(hspace=0.25, wspace=0.25)
        fig.show()

    def _random_sample(self):
        """Generate uniform random sample for initial iteration"""
        for ii in range(self._num_samp):
            pt  = np.random.rand(self._num_dim)*self._param_rng + self._param_min
            self._queue.append(pt)

    def _neighborhood_sample(self):
        """Generate random samples in best Voronoi polygons"""
        
        vv = np.array([x['param'] for x in self._sample])
        vv = (vv - self._param_min)/self._param_rng # normalize
        
        for ii in range(self._num_samp):
            
            # get starting point and all other points as arrays
            kk = ii % self._num_resamp  # index of start point            
            vk = vv[kk,:]
            vj = np.delete(vv, kk, 0)
            xx = vk.copy()
            
            # get initial distance to ith-axis (where i == 0)
            d2ki = 0.0
            d2ji = np.sum(np.square(vj[:,1:] - xx[1:]), axis=1)
            
            # random step within voronoi polygon in each dimension
            for ii in range(self._num_dim):
                
                # find limits of voronoi polygon
                xji = 0.5*(vk[ii] + vj[:,ii] + (d2ki - d2ji)/(vk[ii] - vj[:,ii]))
                try:
                    low = max(0.0, np.max(xji[xji <= xx[ii]]))
                except ValueError: # no points <= current point
                    low = 0.0
                try:
                    high = min(1.0, np.min(xji[xji >= xx[ii]]))
                except ValueError: # no points >= current point
                    high = 1.0

                # random move within voronoi polygon
                xx[ii] = uniform(low, high)
                
                # update distance to next axis
                if ii < (self._num_dim - 1):
                    d2ki += (np.square(vk[ii  ] - xx[ii  ]) - 
                             np.square(vk[ii+1] - xx[ii+1]))
                    d2ji += (np.square(vj[:,ii  ] - xx[ii  ]) - 
                             np.square(vj[:,ii+1] - xx[ii+1]))
                    
            # update queue
            xx = xx*self._param_rng + self._param_min # un-normalize
            self._queue.append(xx)
    
    def _validate_args(self):
        """Check argument types, throw informative exceptions"""
        # # objective
        if not callable(self._objective):
            raise TypeError('"objective" must be a callable')
        # # limits 
        if not isinstance(self._limits, list):
            raise TypeError('"limits" must be a list')
        for lim in self._limits:
            if not isinstance(lim, tuple) or len(lim) != 2:
                raise TypeError('"limits" elements must be length-2 tuples')
            if lim[1] <= lim[0]:
                raise ValueError('"limits" elements must be increasing')
        # # num_samp
        if int(self._num_samp) != self._num_samp:
            raise TypeError('"num_samp" must be an integer')
        if self._num_samp < 1:
            raise ValueError('"num_samp" must be positive')
        # # num_resamp
        if int(self._num_resamp) != self._num_resamp: 
            raise TypeError('"num_resamp" must be an integer')
        if self._num_resamp < 1:
            raise ValueError('"num_resamp" must be positive')    
        if self._num_resamp > self._num_samp:
            raise ValueError('"num_resamp must be <= "num_samp"')
        # # maximize
        if not isinstance(self._maximize, bool):
            raise TypeError('maximize must be boolean: True or False')
        # # verbose
        if not isinstance(self._verbose, bool):
            raise TypeError('verbose must be boolean: True or False')

    def __repr__(self):
        try:
            out = '{}(iteration={}, samples={}, best={:.6e})'.format(
                self.__class__.__name__,
                self._iter,
                len(self._sample),
                self._sample[0]['result'])
        except IndexError:
            out = '{}(iteration=0, samples=0, best=None)'.format(self.__class__.__name__)
        return out


def demo_search(ndim=2, nsamp=10, nresamp=5, niter=100):
    """
    Run demonstration using Rosenbrock objective function, plot results
    
    Arguments:
        ndim: number of dimensions in ND-Rosenbrock function
        nsamp: number of samples for each iteration
        nresamp: number of "best" regions to re-sample for next iteration
        niter: number of iterations to run
    """
    srch = Searcher(
        objective=rosenbrock,
        limits=[(-1.5, 1.5) for _ in range(ndim)],
        num_samp=nsamp,
        num_resamp=nresamp,
        maximize=False,
        verbose=True
        )
    srch.update(niter)
    srch.plot()
    return srch
