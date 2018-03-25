# -*- coding: utf-8 -*-
"""
Neighborhood algorithm direct-search optimization
"""

class Searcher():
    
    def __init__(self):
        """
        Neighborhood algorithm direct-search optimization
        
        Arguments:
        
        References:
            Sambridge, M. (1999). Geophysical inversion with a neighbourhood
            algorithm - I. Searching a parameter space. Geophysical Journal
            International, 138(2), 479–494.
        """
        raise NotImplementedError


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
    Run demonstration and plot results
    
    Arguments:
    """
    raise NotImplementedError
    