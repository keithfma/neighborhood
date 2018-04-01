"""
Reference functions for demonstration and testing
"""


def rosenbrock(xx):
    """
    Rosenbrock ND objective function, a common optimization performance test
    
    Arguments:
        xx: 1D numpy array, coordinates at which to evaluate the N-D Rosenbrock
            test function
    
    References: 
        + Rosenbrock, H. H. (1960). An Automatic Method for finding the Greatest
          or Least Value of a Function. The Computer Journal, 3, 175–84.
        + https://en.wikipedia.org/wiki/Rosenbrock_function
    """
    val = 0
    for ii in range(len(xx)-1):
        t1 = xx[ii+1] - xx[ii]*xx[ii]
        t2 = 1 - xx[ii]
        val += 100*t1*t1 + t2*t2
    return val
