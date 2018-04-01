# -*- coding: utf-8 -*-
"""
Unit tests for neighborhood algorithm direct-search optimization
"""

import pytest
from .search import Searcher
from .reference import rosenbrock
from collections import defaultdict
import numpy as np


LIMITS= (-1.5, 1.5)
NUM_DIM = 4
NUM_SAMP = 10
NUM_RESAMP = 5


@pytest.fixture
def srch():
    """Returns a new Searcher object for Rosenbrock test function"""
    return Searcher(
        objective=rosenbrock,
        limits=[LIMITS for _ in range(NUM_DIM)],
        num_samp=NUM_SAMP,
        num_resamp=NUM_RESAMP,
        maximize=False,
        verbose=False
        )

    
def test_sampling(srch):
    """Sampler should choose NUM_SAMP points in best NUM_RESAMP nbrhoods"""
    for ii in range(3):
        samp = srch.sample
        # TODO: confirm expected num samples
        # TODO: confirm curr iter samples best nbrhoods in prev iter
        srch.update(1)


def test_monotonic(srch):
    """Best objective function should be monotonically improving"""
    srch.update(250)
    for rec in srch.sample:
        pass  
