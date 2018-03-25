# -*- coding: utf-8 -*-
"""
Neighborhood algorithm direct-search optimization
"""

import logging
from copy import deepcopy
from collections import namedtuple
from random import uniform
import numpy as np
from pdb import set_trace


logger = logging.Logger(__name__)


class Searcher():
    
    def __init__(self, na_objective, na_num_samp, na_num_resamp, na_minimize=True, **kwargs):
        """
        Neighborhood algorithm direct-search optimization
        
        Arguments:
            na_objective: callable returning a scalar misfit value, inputs are
                scalar keyword arguments, the paraameter names and limits are
                set by additional keyword arguments to this constructor
                (captured by **kwargs).
            na_num_samp: int, number of random samples taken at each iteration.
            na_num_resamp: int, number of best Voronoi polygons sampled at
                each iteration.
            na_minimize: boolean, set True to minimize the objective function,
                or false to maximize it.
            **kwargs: objective function parameter limits, each provided as
                name=(min_val, max_val), converted internally to a dictionary.
                Parameter names *must not* be the same as the input arguments
                to this constructor -- they are prefixed with "na_" to make
                collisions less likely.
        
        References:
            Sambridge, M. (1999). Geophysical inversion with a neighbourhood
            algorithm - I. Searching a parameter space. Geophysical Journal
            International, 138(2), 479–494.
        """
        # check input types
        # # na_objective
        if not callable(na_objective):
            raise TypeError('"na_objective" must be a callable')
        # # na_num_samp
        if int(na_num_samp) != na_num_samp:
            raise TypeError('"na_num_samp" must be an integer')
        if na_num_samp < 1:
            raise ValueError('"na_num_samp" must be positive')
        # # na_num_resamp
        if int(na_num_resamp) != na_num_resamp: 
            raise TypeError('"na_num_resamp" must be a positive integer')
        if na_num_resamp < 1:
            raise ValueError('"na_num_resamp" must be positive')    
        if na_num_resamp > na_num_samp:
            raise ValueError('"na_num_resamp must be <= "na_num_samp"')
        # # na_minimize
        if not isinstance(na_minimize, bool):
            raise TypeError('na_minimize must be boolean: True or False')
        # # parameter range limits
        for name, lim in kwargs.items():
            if len(lim) != 2:
                raise TypeError('"{}" limits must have length 2'.format(name))
            if lim[1] <= lim[0]:
                raise ValueError('"{}" limits must be increasing'.format(name))
        
        # populate internal constants, etc
        self.objective = na_objective
        self.num_samp = na_num_samp
        self.num_resamp = na_num_resamp
        self.minimize = na_minimize
        self.num_dim = len(kwargs)
        self.limits = deepcopy(kwargs)
        self._pmin = {n: v[0] for n, v in kwargs.items()}
        self._prng = {n: v[1] - v[0] for n, v in kwargs.items()}
        self.Param = namedtuple('Param', kwargs.keys()) 
        self.population = []
        self.queue = []
        self.iter = 0

    def update(self, max_iter=10, tol=1.0e-3):
        """
        Execute search until iteration limit or improvement tolerance is reached
        
        Arguments:
            max_iter:
            tol:
        """
        # TODO: loop until max iterations or tolerance threshold
        
        # generate new sample (populates queue)
        if not self.population:
            self._random_sample()
        else:
            self._neighborhood_sample()
        
        # execute forward model for all samples in queue
        while self.queue:
            param = self.queue.pop()
            result = self.objective(**param._asdict())
            self.population.append((param, result, self.iter))
        
        # reorder population by misfit
        self.population.sort(key=lambda x: x[1])
        
        
        # update iteration counter
        self.iter += 1
        
    def _random_sample(self):
        """Generate uniform random sample for initial iteration"""
        for ii in range(self.num_samp):
            new = {k: uniform(*v) for k, v in self.limits.items()}
            self.queue.append(self.Param(**new))

    def _neighborhood_sample(self):
        """Generate random samples in best Voronoi polygons"""
        for ii in range(self.num_samp):
            
            # get starting point and all other points as arrays
            kk = ii % self.num_resamp  # index of start point
            vk = np.array(self.population[kk][0])
            vj = np.array([x[0] for j, x in enumerate(self.population) if j != kk])
            set_trace()
            
            # NOTE: distances are always along one dimension -- do I really
            #   need to normalize?
            
    def _param_to_pt(self, param):
        """Convert parameter object to point in normalized parameter space"""
        pt = [(v - self._pmin[n])/self._prng[n] for n, v in param._asdict().items()]
        return np.array(pt)

    # TODO: include a plot of the best objective function value per iteration

    def plot(polygons=False, marginals=False):
        """
        Plot of objective function values for current population
        
        Displays pair-wise plots for all parameter combinations. Optionally,
        can include filled voronoi polygons (use only for *small* populations)
        and marginal histograms for each parameter.
        
        Arguments:
            polygons: Plot filled Voronoi polygons
            marginals: Plot marginal histograms for each parameter
        """
        raise NotImplementedError
        
    def dump(self, filename, clobber=False):
        """
        Save current optimizer state to file as pickle
        
        Arguments:
            filename: string, filename for saved result
            clobber: set True to overwrite existing file, otherwise if
                "filename" exists, a timestamp is appended to make it unique
        """
        raise NotImplementedError
    
    @classmethod
    def load(cls, filename):
        """
        Load optimizer from pickle file
        
        Arguments:
            filename: string filename of saved result to load
        
        Returns: Searcher object loaded from file
        """
        raise NotImplementedError

    # def __repr__(self): # TODO
    
    # def __str__(self): # TODO


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
        na_num_samp=10,
        na_num_resamp=2,
        na_minimize=True,
        xx=(0, 2),  # param to objective function
        yy=(0,2)    # param to objective function
        )

    # debug
    return srch