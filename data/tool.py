#!/usr/bin/python
# -*- coding: UTF-8 -*-


import numpy
import timeit
import theano.tensor as T
import six.moves.cPickle as pickle

from mlp import MLP

def get_pickling_errors(obj,seen=None):
    if seen == None:
          seen = []
    try:
         state = obj.__getstate__()
    except AttributeError:
         return
         return
    if isinstance(state,tuple):
         if not isinstance(state[0],dict):
             state=state[1]
         else:
             state=state[0].update(state[1])
    result = {}    
    for i in state:
         try:
             pickle.dumps(state[i],protocol=2)
         except pickle.PicklingError:
             if not state[i] in seen:
                 seen.append(state[i])
                 result[i]=get_pickling_errors(state[i],seen)
    return result


x = T.matrix('x')    
rng = numpy.random.RandomState(1234)
    
mlp = MLP(rng,x,10,10,10)
get_pickling_errors(mlp)
