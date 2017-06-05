#!/usr/bin/python
# -*- coding: UTF-8 -*-


import numpy
import timeit
import six.moves.cPickle as pickle

import theano
import theano.tensor as T
from   mHiddenLayer import HiddenLayer
import logisRegression as util
from logisRegression import LogisticRegression

class MLP(object):
    
    def __init__(self, rng, input, n_in,n_hidden, n_out):
        
        self.hiddenLayer = HiddenLayer(rng,input, n_in=n_in, n_out = n_hidden,activation= T.tanh)
        
        self.logicRegressionLayer = LogisticRegression(input=self.hiddenLayer.output, n_in = n_hidden,n_out= n_out,)
        
        
        self.L1 = ( abs(self.hiddenLayer.W).sum() + abs(self.logicRegressionLayer.W).sum() )
        
        self.L2_sqr = (self.hiddenLayer.W**2).sum() + (self.logicRegressionLayer.W ** 2).sum()
        
        self.negative_log_likelihood = (self.logicRegressionLayer.negative_log_likelihood )
        
        self.errors = self.logicRegressionLayer.errors
        
        self.params = self.hiddenLayer.params + self.logicRegressionLayer.params
        
        self.input = input
        
        
## train mlp        
def train_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000, 
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
    
    rng = numpy.random.RandomState(1234)
    
    start = index * batch_size
    end  =  (index+1) * batch_size
    classifier = MLP(rng, x, n_in = 28*28, n_hidden=n_hidden, n_out=10)
    
    cost = ( classifier.negative_log_likelihood(y) + L1_reg* classifier.L1 + L2_reg*classifier.L2_sqr)
    
    test_model = theano.function(inputs=[index],outputs=classifier.errors(y), 
                                 givens={ x: test_set_x[start:end], y: test_set_y[start:end] } )
    
    validate_model = theano.function(inputs=[index], outputs= classifier.errors(y),
                                     givens={ x:validate_set_x[start:end], y:validate_set_y[start:end] })
    
    gparams = [ T.grad(cost, param) for param in classifier.params ]   
                
    updates = [( param, param - learning_rate* gparam) for param,gparam in zip(classifier.params , gparams )]
    
    train_model = theano.function( inputs=[index], outputs=cost, updates = updates, 
                                   givens={ x:train_set_x[start:end], y:train_set_y[start:end]})
    
    """
        Start Train model
    """
    
    print('....start training....')    
    
        ### Start train model                                            
    patience = 5000
    patience_increase = 2
    improvement_threshold = 0.995

    validation_frequency = min( n_train_batches, patience/2 )
    
    print('validation_frequency %f ' % validation_frequency)
    
    best_validation_loss = numpy.inf
    test_score = 0.
    starttime = timeit.default_timer()

    done_looping = False
    epoch = 0

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1 
        #print('Epoch %i ......' % epoch )
        for minibatch_index in xrange(n_train_batches):
            #iteration number
            
            # train model
            minibatch_avg_cost = train_model( minibatch_index)
            
            iter = (epoch -1 ) * n_train_batches + minibatch_index

            if (iter+1)%validation_frequency == 0 :
                validation_losses = [validate_model(i) for i in xrange(n_validate_batches)]
                #avg loss
                this_validation_loss = numpy.mean( validation_losses)

                print( 'epoch %i, minibatch %i/%i, validation_loss %f %%'%
                        (epoch, minibatch_index+1,  n_train_batches, this_validation_loss*100 ))


                if this_validation_loss < best_validation_loss:
                    if this_validation_loss< best_validation_loss*improvement_threshold:
                        patience = max( patience, iter*patience_increase)
                    best_validation_loss = this_validation_loss
                    
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]    
                    test_score  = numpy.mean(test_losses)

                    print(('epoch %i, minibatch %i/%i, test error %f %%. Update best validation loss.') %
                            (epoch, minibatch_index+1, n_train_batches, test_score*100 ))

                    #with open('best_model.pkl','w') as f:
                    #    pickle.dump( classifier,f )
            if iter > patience:
                print('End looping. Cause iter %i > patience %i.' % (iter,patience))
                done_looping = True
                break
    ##end while
    
    ##compute time used
    endtime = timeit.default_timer()

    ##finish validation 
    print( ('Optimization complete with best validation sore %f %%, test score %f %%') % 
            (best_validation_loss*100, test_score*100))

    print( 'Code run for epoch %d, with %f epoch/sec' % (epoch, 1.0*epoch/(endtime-starttime)))
    
    print >> sys.stderr, ('Thr code for file '+  os.path.split(__file__)[1] +'ran for %.1fs'%((endtime-starttime)))    


if __name__ =="__main__":
    train_mlp()
    
        