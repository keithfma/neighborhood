# -*- coding: utf-8 -*-
"""
Unit tests for neighborhood algorithm direct-search optimization
"""

import pytest
from .search import Searcher
from .reference import rosenbrock
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
    NUM_ITER = 10
    srch.update(1)
    for ii in range(1, NUM_ITER):
        srch.update(1)
        samp = srch.sample_dataframe
        # confirm expected num samples
        assert(samp.shape[0] == (ii + 1)*NUM_SAMP)
        # confirm curr iter samples best nbrhoods in prev iter
        prev = samp[samp['iter'] < ii].sort_values(by='result')
        prev_pts = prev.drop(['iter', 'result'], axis=1)
        curr = samp[samp['iter'] == ii] 
        curr_pts = curr.drop(['iter', 'result'], axis=1)
        for ii in range(NUM_SAMP):
            pt = curr_pts.iloc[ii]
            dist2 = ((pt - prev_pts)**2).sum(axis=1)
            nearest_rank = dist2.values.argmin()
            assert(nearest_rank < NUM_RESAMP)


def test_monotonic(srch):
    """Best objective function should be monotonically improving"""
    NUM_ITER = 50
    srch.update(NUM_ITER)
    samp = srch.sample_dataframe
    prev_best = float('inf')
    for ii in range(NUM_ITER):
        curr_best = min(samp[samp['iter'] <= ii]['result'])
        assert(curr_best <= prev_best)
        prev_best = curr_best
