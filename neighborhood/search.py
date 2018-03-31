# -*- coding: utf-8 -*-
"""
Neighborhood algorithm direct-search optimization
"""

from copy import deepcopy
import collections
from random import uniform
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Searcher():
    
    def __init__(self, objective, limits, num_samp, num_resamp, maximize=False, verbose=True):
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
        self._maximize = maximize
        self._verbose = verbose
        self._validate_args()

        # init constants and working vars
        self._num_dim = len(limits)
        self._param_min = [x[0] for x in limits]
        self._param_max = [x[1] for x in limits]
        self._param_rng = [x[1]-x[0] for x in limits]
        self._sample = np.empty(shape=(0, self.num_dim+1), dtype=np.double)
        self._queue = []
        self._iter = 0
        
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
            if lim[1] >= lim[0]:
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

    # def as_dataframe(self):
    #     """Return sampled population as Pandas dataframe"""
    #     pop = [{'objective': x['result'], **(x['param']._asdict())}
    #             for x in self.population] 
    #     return pd.DataFrame(pop)
        
    def update(self, max_iter=10, tol=1.0e-3):
        """
        Execute search algorithm for specified number of iterations
        
        Arguments:
            num_iter: int, number of iterations to run
        """
        
        for ii in range(max_iter):
            
            # generate new sample (populates queue)
            if not self.population:
                self._random_sample()
            else:
                self._neighborhood_sample()
                        
            # execute forward model for all samples in queue
            while self.queue:
                param = self.queue.pop()
                result = self.objective(**param._asdict())
                self.population.append({
                    'param': param,
                    'result': result,
                    'iter': self.iter
                    })
            
            # prepare for next iteration
            self.population.sort(key=lambda x: x['result'], reverse=self.maximize)
            self.iter += 1
            if self.verbose:
                print('iter: {}, pop size: {}, objective: {}'.format(self.iter,
                      len(self.population), self.population[0]['result']))

    def _random_sample(self):
        """Generate uniform random sample for initial iteration"""
        for ii in range(self.num_init):
            new = {k: uniform(*v) for k, v in self.limits.items()}
            self.queue.append(self.Param(**new))

    def _neighborhood_sample(self):
        """Generate random samples in best Voronoi polygons"""
        
        vv = np.array([x['param'] for x in self.population])
        vv = (vv - self.min_param)/self.rng_param # normalize
        
        for ii in range(self.num_samp):
            
            # get starting point and all other points as arrays
            kk = ii % self.num_resamp  # index of start point            
            vk = vv[kk,:]
            vj = np.delete(vv, kk, 0)
            xx = vk.copy()
            
            # get initial distance to ith-axis (where i == 0)
            d2ki = 0.0
            d2ji = np.sum(np.square(vj[:,1:] - xx[1:]), axis=1)
            
            # random step within voronoi polygon in each dimension
            for ii in range(self.num_dim):
                
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
                if ii < (self.num_dim - 1):
                    d2ki += (np.square(vk[ii  ] - xx[ii  ]) - 
                             np.square(vk[ii+1] - xx[ii+1]))
                    d2ji += (np.square(vj[:,ii  ] - xx[ii  ]) - 
                             np.square(vj[:,ii+1] - xx[ii+1]))
                    
            # update queue
            xx = xx*self.rng_param + self.min_param # un-normalize    
            self.queue.append(self.Param(*xx))

    def plot(self):
        """
        Display pair-plots of objective function values for current population
        """
        if self.num_dim == 2:
            df = self.as_dataframe()
            df.plot.scatter(
                x=df.columns[1],
                y=df.columns[2],
                c='objective',
                colormap=plt.get_cmap('plasma')
                )
            plt.show()        
        else:
            raise NotImplementedError('Plotting not yet implemented in > 2D')
    
    def __repr__(self):
        try:
            out = '{}(pop_size={}, best={:.6e})'.format(
                self.__class__.__name__, len(self.population), 
                self.population[0]['result'])
        except IndexError:
            out = '{}(pop_size=0)'.format(self.__class__.__name__)
        return out
    

# TODO: generalize to higher dimensions
#   see: https://en.wikipedia.org/wiki/Rosenbrock_function
def _rosenbrock(xx, yy, aa=1.0, bb=100.0):
    """
    Rosenbrock 2D objective function, a common optimization performance test
    
    Note: minimum is at xx = aa, yy = aa**2
    
    Arguments:
        xx, yy: Scalar, float, coordinates at which to evaluate the Rosenbrock
            test function
        aa, bb: Scalar, float, parameters to the Rosenbrock function
    
    References: 
        Rosenbrock, H. H. (1960). An Automatic Method for finding the Greatest
        or Least Value of a Function. The Computer Journal, 3, 175–84.
    """
    t1 = aa - xx
    t2 = yy - xx*xx
    return t1*t1 + bb*t2*t2


def demo_search():
    """
    Run demonstration using Rosenbrock objective function, plot results
    
    Arguments:
    """
    srch = Searcher(
        na_objective=_rosenbrock,
        na_num_samp=900,
        na_num_resamp=450,
        na_num_init=900,
        na_maximize=False,
        xx=(-1.5, 1.5),  # param to objective function
        yy=(-0.5, 3.0)   # param to objective function
        )
    srch.update(20)
    srch.plot()
    return srch
