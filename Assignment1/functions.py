import numpy as np


def EvaluateClassifier(X, W, b): # Excercise 1.4, no need to transpose x!
    # s = W.dot(x) + b.T # Vec is dim x 1, W is K x dim, b is K x 1
    s = W @ X + b
    return softmax(s)

def ComputeCost(X, Y, W, b, lamb): # Excercise 1.5
    J = 0
    P = EvaluateClassifier(X, W, b)
    # P = np.zeros((X.shape[0], 10))
    # for i in range(len(X)):
    #     p = EvaluateClassifier(X[i], W, b)
    #     p_y = -np.dot(Y[i], np.log(p))
    #     J += p_y
    #     P[i] = p.ravel()
    P_t = P.T
    for i in range(len(P_t)):
        J += -np.dot(Y[i], np.log(P_t[i]))
    J /= len(X[0]) # Divide by dimensionality
    loss = J # For documentation
    J += lamb * np.sum(np.power(W,2)) # WTerm
    
    return J, P
    
    return J, P

def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def LoadBatch(filename):
	""" Copied from the dataset website """
	import pickle
	with open("data/data_batch_1", 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

def ComputeGradsNum(X, Y, P, W, b, lamda, h):
	""" Converted from matlab code """
	no 	= 	W.shape[0]
	d 	= 	X.shape[0]

	grad_W = np.zeros(W.shape)
	grad_b = np.zeros((no, 1))

	c,_ = ComputeCost(X, Y, W, b, lamda)
	print(c)
	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] += h
		c2,_ = ComputeCost(X, Y, W, b_try, lamda)
		grad_b[i] = (c2[0][0]-c[0][0]) / h

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] += h
			c2,_ = ComputeCost(X, Y, W_try, b, lamda)
			grad_W[i,j] = (c2[0][0]-c[0][0]) / h

	return [grad_W, grad_b]

def ComputeGradsNumSlow(X, Y, P, W, b, lamda, h):
	""" Converted from matlab code """
	no 	= 	W.shape[0]
	d 	= 	X.shape[0]

	grad_W = np.zeros(W.shape)
	grad_b = np.zeros((no, 1))
	
	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] -= h
		c1 = ComputeCost(X, Y, W, b_try, lamda)

		b_try = np.array(b)
		b_try[i] += h
		c2 = ComputeCost(X, Y, W, b_try, lamda)
		grad_b[i] = (c2[0]-c1[0]) / (2*h)

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] -= h
			c1 = ComputeCost(X, Y, W_try, b, lamda)

			W_try = np.array(W)
			W_try[i,j] += h
			c2 = ComputeCost(X, Y, W_try, b, lamda)

			grad_W[i,j] = (c2[0]-c1[0]) / (2*h)

	return [grad_W, grad_b]

def montage(W):
	""" Display the image for each label in W """
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(2,5)
	for i in range(2):
		for j in range(5):
			im  = W[i*5+j,:].reshape(32,32,3, order='F')
			sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
			sim = sim.transpose(1,0,2)
			ax[i][j].imshow(sim, interpolation='nearest')
			ax[i][j].set_title("y="+str(5*i+j))
			ax[i][j].axis('off')
	plt.show()

# def save_as_mat(data, name="model"):
# 	""" Used to transfer a python model to matlab """
# 	import scipy.io as sio
# 	sio.savemat(name+'.mat',{name:b})