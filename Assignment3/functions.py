import numpy as np


def EvaluateClassifier(X, W1, W2, b1, b2): # Returns the final P values and the intermidiary activation values (H)
    s1 = CalcS(X, W1, b1) # s1 is mxD, TODO: Check if we should diagnoalize b1
    H = np.maximum(0, s1) # H is mxD, , returns the element-wise max of s1 and 0
    s2 = CalcS(H, W2, b2) # s2 is mxC
    P = softmax(s2)
    return P, H

def CalcS(X, W, b):
    return W @ X + b

def ComputeCost(X, Y, W1, W2, b1, b2, lamb): 
    J = 0
    P,_ = EvaluateClassifier(X, W1, W2, b1, b2)
    P_t = P.T
    for i in range(len(P_t)):
        J += -np.dot(Y[i], np.log(P_t[i]))
    J /= len(X[0]) # Divide by dimensionality
    loss = J # For documentation
    J += lamb * (np.sum(np.power(W1,2)) + np.sum(np.power(W2,2))) # WTerm
    
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



def compute_grads_num_2(data, labels, weights, bias, lmb, h):
    grad_weights = list()
    grad_bias = list()

    c = ComputeCost(data, labels, weights, bias, lmb)

    for j in range(len(bias)):
        grad_bias.append(np.zeros(len(bias[j])))
        for i in range(len(grad_bias[j])):
            b_try = list()
            [b_try.append(np.copy(x)) for x in bias]
            b_try[j][i] = b_try[j][i] + h
            c2 = ComputeCost(data, labels, weights, b_try, lmb)
            grad_bias[j][i] = (c2 - c) / h

    for j in range(len(weights)):
        grad_weights.append(np.zeros(weights[j].shape))
        for i in range(grad_weights[-1].shape[0]):
            for k in range(grad_weights[-1].shape[1]):
                w_try = list()
                [w_try.append(np.copy(x)) for x in weights]
                w_try[j][i, k] = w_try[j][i, k] + h
                c2 = ComputeCost(data, labels, w_try, bias, lmb)
                grad_weights[j][i, k] = (c2 - c) / h

    return grad_weights, grad_bias