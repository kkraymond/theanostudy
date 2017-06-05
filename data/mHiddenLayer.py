#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
    Hidden Layer def
"""

import numpy as np
import theano
import theano.tensor as T

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out,W=None,b=None, activation=T.tanh):
        self.input = input
        
        # init W, b
        if W is None:
            bound = np.sqrt( 6.0/(n_in + n_out )) ;
            Wr = rng.uniform( low = -bound, high = bound,size=(n_in, n_out))
            Wv = np.asarray( Wr, dtype = theano.config.floatX)
            
            W = theano.shared( value = Wv, name='W', borrow=True)
        if b is None:
            bv = np.zeros( (n_out,), dtype=theano.config.floatX )    
            b = theano.shared(value= bv, name='b',borrow=True)
        self.W = W
        self.b = b    
        
        lin_output = T.dot( input, self.W) + self.b
        self.output = ( lin_output if activation is None 
                        else activation( lin_output )
                      )
        
        self.input = input
        self.params = [self.W, self.b]
        
        
       