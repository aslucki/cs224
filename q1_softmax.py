import numpy as np
import time

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print '%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000)
        return result
    return timed


def softmax(x):
	"""Compute the softmax function for each row of the input x.

	It is crucial that this function is optimized for speed because
	it will be used frequently in later code. You might find numpy
	functions np.exp, np.sum, np.reshape, np.max, and numpy
	broadcasting useful for this task.

	Numpy broadcasting documentation:
	http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

	You should also make sure that your code works for a single
	N-dimensional vector (treat the vector as a single row) and
	for M x N matrices. This may be useful for testing later. Also,
	make sure that the dimensions of the output match the input.

	You must implement the optimization in problem 1(a) of the
	written assignment!

	Arguments:
	x -- A N dimensional vector or M x N dimensional numpy matrix.

	Return:
	x -- You are allowed to modify x in-place
	"""
	orig_shape = x.shape

	if len(x.shape) > 1:
		# Matrix
		max_values = np.max(x, axis=1, keepdims=True)
		numerators = np.exp(x - max_values)
		x = numerators/np.sum(numerators,axis=1, keepdims=True)
	else:
	    #Vector
		max_value = np.max(x)
		numerators = np.exp(x - max_value)
		x = numerators/np.sum(numerators)

	assert x.shape == orig_shape
	return x


def test_softmax_basic():
	"""
	Some simple tests to get you started.
	Warning: these are not exhaustive.
	"""
	print "Running basic tests..."
	test1 = softmax(np.array([1,2]))
	print test1
	ans1 = np.array([0.26894142,  0.73105858])
	assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

	test2 = softmax(np.array([[1001,1002],[3,4]]))
	print test2
	ans2 = np.array([
		[0.26894142, 0.73105858],
		[0.26894142, 0.73105858]])
	assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

	test3 = softmax(np.array([[-1001,-1002]]))
	print test3
	ans3 = np.array([0.73105858, 0.26894142])
	assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

	print "You should be able to verify these results by hand!\n"

@timeit
def test_softmax_big_matrix(x):
    return softmax(x)
	
def test_softmax():
	"""
	Use this space to test your softmax implementation by running:
		python q1_softmax.py
	This function will not be called by the autograder, nor will
	your tests be graded.
	"""
	print "Running your tests..."
	test4 = test_softmax_big_matrix(np.random.rand(100000,100)*10000)
	assert np.allclose(np.sum(test4, axis=1), np.ones(test4.shape[0]), 
		rtol=1e-05, atol=1e-06)
	
	print "Values calculated on a big matrix were correct"

if __name__ == "__main__":
	test_softmax_basic()
	test_softmax()
