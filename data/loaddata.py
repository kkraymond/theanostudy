import cPickle as pickle
import gzip
import  numpy as np
import theano
import theano.tensor as T
#import theano.config




def loadDataSet():
	f = gzip.open('mnist.pkl.gz','rb');
	train_set, validate_set, test_set = pickle.load( f );

	data_x, data_y = train_set;
	print('trainset data_x',len(data_x),type(data_x), data_x.dtype)
	print('trainset data_y',len(data_y))
	f.close();
	return train_set, validate_set, test_set

def shared_dataset( data_xy ):
	data_x, data_y = data_xy
	shared_x = theano.shared( data_x)  #np.asarray(data_x, dtype=theano.config.floatX))
	shared_y = theano.shared( data_y)  #np.asarray(data_y, dtype=theano.config.floatX))

	return shared_x, T.cast( shared_y, 'int32')


##>>>>>>>>>>>>>>>>>>>>>>
train_set, validate_set, test_set = loadDataSet()

train_set_x, train_set_y = shared_dataset( train_set)
alidate_set_x, alidate_set_y = shared_dataset( validate_set)
test_set_x, test_set_y = shared_dataset( test_set)


batch_size = 10
batch_index = 3
data = train_set_x[ (batch_index -1) * batch_size : (batch_index)* batch_size ]
label = train_set_y[  (batch_index -1) * batch_size : (batch_index)* batch_size ]

#for d, t  in zip(data,label):
#	print('train data:',d,t)