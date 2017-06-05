
__docformat__ = 'restructedtext en'


import six.moves.cPickle as pickle
import gzip, os, sys, timeit

import numpy
import numpy as np
import theano 
import theano.tensor as T


def loadDataSet(datasetFile):
	f = gzip.open( datasetFile ) #,'rb');
	train_set, validate_set, test_set = pickle.load( f );

	data_x, data_y = train_set;
	print('trainset data_x',len(data_x),type(data_x), data_x.dtype)
	print('trainset data_y',len(data_y))
	f.close();
	return train_set, validate_set, test_set

def shared_dataset( data_xy,borrow=True ):
	data_x, data_y = data_xy
	shared_x = theano.shared( np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
	shared_y = theano.shared( np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)

	return shared_x, T.cast( shared_y, 'int32')

def process_dataset( datasetFile ):
	train_set, validate_set, test_set = loadDataSet(datasetFile)

	train_set_x, train_set_y = shared_dataset( train_set)
	validate_set_x, validate_set_y = shared_dataset( validate_set)
	test_set_x, test_set_y = shared_dataset( test_set)

	rval = [(train_set_x, train_set_y), (validate_set_x, validate_set_y), (test_set_x, test_set_y) ]
	return rval


"""
 Class logisitic Regression
"""

class LogisticRegression(object):

	# init
	def __init__(self, input, n_in, n_out):

		#n_in, n_out = 6, 4  # input size, output size

		W0 = np.zeros( (n_in, n_out),dtype=theano.config.floatX)
		b0 = np.zeros( (n_out,), dtype=theano.config.floatX) 

		self.W = theano.shared( W0, 'W',borrow=True)
		self.b = theano.shared( b0, 'b',borrow=True)

		

		y0 = T.dot( input, self.W ) + self.b
		self.p_y_given_x = T.nnet.softmax( y0 )
		self.y_pred = T.argmax( self.p_y_given_x, axis=1)

		self.params = [self.W,self.b]
		self.input = input


	# NNL
	def negative_log_likelihood(self,y):
		logP = T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]
		nnl = -T.mean( logP )
		return nnl


	#Errors ---------------------
	def errors(self, y):
		if y.ndim!=self.y_pred.ndim:
			raise TypeError('Shape not same: y and y_pred','y',y.type,'y_pred',y_pred.type)

		if y.dtype.startswith('int'):
			return T.mean( T.neq(self.y_pred, y))
		else:
			raise NotImplementedError()	

	
# SGD algorithm
def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000, dataset = 'mnist.pkl.gz', batch_size=600 ):
	datasets = process_dataset( dataset )

	train_set_x, train_set_y = datasets[0]
	validate_set_x, validate_set_y = datasets[1]
	test_set_x, test_set_y = datasets[2]

	n_train_batches    = train_set_x.get_value(borrow=True).shape[0]/batch_size
	n_validate_batches = validate_set_x.get_value(borrow=True).shape[0]/batch_size
	n_test_batches     = test_set_x.get_value(borrow=True).shape[0]/batch_size

	print( '......build the model')

	print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
	print('LogisticRegression')

	

	x = T.matrix('x')
	y = T.ivector('y')

	classifier = LogisticRegression(input=x, n_in=28*28,n_out=10)
	cost = classifier.negative_log_likelihood( y ) 

	index = T.lscalar()  #index of minibatch
	start = index * batch_size
	end   = (index+1) * batch_size
	validate_model = theano.function(inputs=[index],outputs=classifier.errors( y ), 
									givens = { x:validate_set_x[start:end] , y: validate_set_y[start:end]} )

	test_model = theano.function(inputs=[index], outputs=classifier.errors(y),
									givens= {x: test_set_x[start:end], y:test_set_y[start:end]})

	g_W = T.grad(cost=cost, wrt=classifier.W)
	g_b = T.grad(cost=cost, wrt=classifier.b)

	updates = [ (classifier.W, classifier.W - learning_rate * g_W),
				(classifier.b, classifier.b - learning_rate * g_b) ]

	train_model = theano.function([index],cost,updates=updates, 
									givens = {x:train_set_x[start:end], y:train_set_y[start:end]})

	

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

					with open('best_model.pkl','w') as f:
						pickle.dump( classifier,f )
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


def trainModel(learning_rate=0.13, n_epochs=1000, dataset = 'mnist.pkl.gz', batch_size=600 ):
	return sgd_optimization_mnist(learning_rate, n_epochs, dataset,batch_size )


def pridict():

	classifier = pickle.load( open('best_model.pkl'))

	predict_model = theano.function([classifier.input], classifier.y_pred)

	dataset = 'mnist.pkl.gz'
	datasets = process_dataset( dataset )

	test_set_x, test_set_y =  datasets[2]

	test_set_x  = test_set_x.get_value()

	predict_values = test_model( test_set_x[:10])

	print( 'predict_values of the leading 10 examples')
	print( predict_values)

if __name__ == '__main__':
	trainModel(learning_rate=0.13, n_epochs=1000, dataset = 'mnist.pkl.gz', batch_size=200 )	

