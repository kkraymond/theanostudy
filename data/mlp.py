#!/usr/bin/python
# -*- coding: UTF-8 -*-


import numpy as np

import theano
import theano.tensor as T
import mHiddenLayer
import logisRegression as util
from logisRegression import LogisticRegression

class MLP(object):
    
    def __init__(self, rng, input, n_in,n_hidden, n_out):
        
        self.hiddenLayer = HiddenLayer(rng,input, n_in=n_in, n_out = n_hidden,activation= T.tanh)
        
        self.logicRegressionLayer = LogisticRegression(input=self.hiddenLayer.output, n_in = n_hidden,n_out= n_out,)
        
        
        self.L1 = ( abs(self.hiddenLayer.W).sum() + abs(self.logicRegressionLayer.W).sum() )
        
        self.L2_sqr = (self.hiddenLayer.W**2).sum() + (self.logicRegressionLayer.W ** 2).sum()
        
        self.negative_log_likelihood = (self.logicRegressionLayer.negative_log_likelihood )
        
        self.errors = self.errors
        
        self.params = self.hiddenLayer.params + self.logicRegressionLayer.params
        
        self.input = input
        
    def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000, 
                    dataset='mnist.pkl.gz', batch_size=20, n_hidden=500):
         
        datasets = util.process_dataset( dataset )
         
        train_set_x, train_set_y = datasets[0]
        validate_set_x, validate_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]
    
        n_train_batches    = train_set_x.get_value(borrow=True).shape[0]/batch_size
        n_validate_batches = validate_set_x.get_value(borrow=True).shape[0]/batch_size
        n_test_batches     = test_set_x.get_value(borrow=True).shape[0]/batch_size

        print('...building the model')
        index = T.lscalar()
        x = T.matrix('x')
        y = T.ivector('y')
        
        rng = np.random.RandomState(1234)
        
        classifier = MLP(rng, x, n_in = 28*28, n_hidden, n_out=10)
        
        cost = ( classifier.negative_log_likelihood(y) + L1_reg* classifier.L1 + L2_reg*classifier.L2_sqr)
        
        test_model = theano.function()
            
        