import numpy as np

def linearRegCostFunction(X, y, theta, lamb):
	"""
	LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
	regression with multiple variables
	[J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
	cost of using theta as the parameter for linear regression to fit the 
	data points in X and y. Returns the cost in J and the gradient in grad
	"""

	#Initialize some useful values
	m = len(y) # number of training examples
	# ====================== YOUR CODE HERE ======================
	# Instructions: Compute the cost and gradient of regularized linear 
	#               regression for a particular choice of theta.
	#
	#               You should set J to the cost and grad to the gradient.
	#
	h=np.dot(X,theta)

	theta_zero=np.concatenate(([0],theta[1:]))

	J=(1/(2.0*m))*np.sum((h-y)**2)+(lamb/(2.0*m))*np.sum(theta_zero**2)

	return J
	#===============================================================

def linRegGrad(X,y,theta,lamb):
	m=len(y)
	h=np.dot(X,theta)
	theta_zero=np.concatenate(([0],theta[1:]))	
	grad=(1.0/m)*(np.dot(np.transpose(X),h-y)+lamb*theta_zero)	
	return grad

#a function that takes in theta as an input, and considers the tuple to be X, y, and lamb
def linGrad(theta,*args):
	X, y, lamb=args
	return linRegGrad(X, y, theta, lamb)

def linCost(theta,*args):
	X, y, lamb=args
	return linearRegCostFunction(X, y, theta, lamb)	

def trainLinearReg(X,y,lamb):
	from scipy import optimize
	initial_theta=np.zeros(X.shape[1])
	in_args=(X,y,lamb)
	theta=optimize.fmin_cg(linCost, initial_theta, fprime=linGrad, args=in_args, maxiter=200)
	return theta

def learningCurve(X, y, Xval, yval, lamb):
	import random
	n_samp=50

	m=len(y)
	#create an array of training and test validation errors
	error_train=np.zeros(m)
	error_val=np.zeros(m)
	error_train_samp=np.zeros(n_samp)
	error_val_samp=np.zeros(n_samp)

	x_index=np.arange(0,m)
	xval_index=np.arange(0,len(yval))
	#take first i training examples in X/y, compute theta, find error
	for i in xrange(1, m+1):
		for t in xrange(0,n_samp):
			x_r=random.sample(x_index,i)
			X_i=X[x_r]
			y_i=y[x_r]

			xval_r=random.sample(xval_index,i)
			Xval_i=Xval[xval_r]
			yval_i=yval[xval_r]

			theta_i=trainLinearReg(X_i, y_i, lamb)
			error_train_samp[t]=linearRegCostFunction(X_i, y_i, theta_i, 0)
			error_val_samp[t]=linearRegCostFunction(Xval_i, yval_i, theta_i, 0)

		error_train[i-1]=np.average(error_train_samp)
		error_val[i-1]=np.average(error_val_samp)	
	return [error_train, error_val]	

def polyFeatures(X,p):
	Xp=np.zeros((np.shape(X)[0],p))
	for i in xrange(0,p):
		Xp[:,i]=X**(i+1)
	return Xp

def featureNormalize(X):
	mu=np.mean(X,axis=0)		
	std=np.std(X,axis=0)
	#now need to apply these to X along the proper dimension
	return ((X-mu)/std, mu, std)

def stackOnes(X):
	m=len(X)
	return np.reshape(np.hstack((np.ones(m),X.flatten('F'))),(m,-1),'F')

def plotFit(min_x, max_x, mu, sigma, theta, p):
	import matplotlib.pyplot as pyplot
	"""
	Plots a learned polynomial regression fit over an existing figure
	"""
	#generate the x vector, normalize it, p it, and stack
	X=np.arange(min_x-15, max_x+25, 0.05)
	X_poly=polyFeatures(X,p)
	X_poly=(X_poly-mu)/sigma
	X_poly=stackOnes(X_poly)

	pyplot.plot(X, np.dot(X_poly,theta), '--', linewidth=2)

def validationCurve(lamb_vec, X, y, Xval, yval):
	"""
	For each value of lambda given in the lamb vec, trains using X/y, and computes error using theta on X/y, Xval/yval
	If you want X/Xval normalized, should pass normalized versions to this function (does not normalize)
	"""
	error_train=np.zeros(lamb_vec.size)
	error_val=np.zeros(lamb_vec.size)
	for i, lamb in enumerate(lamb_vec):
		theta=trainLinearReg(X, y, lamb)
		error_train[i]=linearRegCostFunction(X, y, theta, 0)
		error_val[i]=linearRegCostFunction(Xval, yval, theta, 0)
	return (error_train, error_val)	