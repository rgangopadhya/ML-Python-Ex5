def main():
	from scipy.io import loadmat
	import matplotlib.pyplot as pyplot
	import numpy as np
	from lincost import linearRegCostFunction, linRegGrad, trainLinearReg, \
	learningCurve, polyFeatures, featureNormalize, stackOnes, plotFit, validationCurve

	print 'Loading and Visualizing Data...'
	datafile="ex5data1.mat"
	data = loadmat(datafile, matlab_compatible=True)
	pdata=dict()

	for key in data.keys():
		if key[0]!='_':
			pdata[key]=data[key].squeeze()

	m=len(pdata['X'])
	
	pyplot.figure(0)
	pyplot.plot(pdata['X'],pdata['y'],'rx',markersize=10, linewidth=1.5)
	pyplot.xlabel('Change in water level (x)')
	pyplot.ylabel('Water flowing out of the dam (y)')

	theta=np.array([1,1])
	X=stackOnes(pdata['X'])
	print X.shape
	J=linearRegCostFunction(X, pdata['y'], theta, 1)
	grad=linRegGrad(X,pdata['y'],theta,1)

	print 'Cost at theta=[1 1]:\n (this value should be about 303.993192)'
	print J
	print 'Gradient at theta=[1 1]'
	print grad

	lamb=0
	theta=trainLinearReg(X,pdata['y'],lamb)
	print theta
	pyplot.plot(pdata['X'],np.dot(X,theta),'--',linewidth=2)
	
	lamb=0
	Xval=stackOnes(pdata['Xval'])
	error_train, error_val=learningCurve(X, pdata['y'], Xval, pdata['yval'], lamb)

	pyplot.figure(1)
	pyplot.plot(np.arange(1,m+1), error_train, np.arange(1,m+1), error_val)

	p=8

	X_poly=polyFeatures(pdata['X'], p)
	X_poly_n, mu, std = featureNormalize(X_poly)
	X_poly_no=stackOnes(X_poly_n)

	X_poly_test=polyFeatures(pdata['Xtest'], p)
	X_poly_test_n=(X_poly_test-mu)/std
	X_poly_test_no=stackOnes(X_poly_test_n)

	X_poly_val=polyFeatures(pdata['Xval'], p)
	X_poly_val_n=(X_poly_val-mu)/std
	X_poly_val_no=stackOnes(X_poly_val_n)

	lamb=0.01
	theta=trainLinearReg(X_poly_no, pdata['y'], lamb)
	pyplot.figure(2)
	pyplot.plot(pdata['X'], pdata['y'], 'rx', markersize=10, linewidth=1.5)
	plotFit(np.min(X), np.max(X), mu, std, theta, p)
	pyplot.xlabel('Change in water level (x)')
	pyplot.ylabel('Water flowing out of the dam (y)')
	pyplot.title('Polynomial Regression Fit (lambda = %s)'.format(lamb))

	pyplot.figure(3)
	error_train, error_val=learningCurve(X_poly_no, pdata['y'], X_poly_val_no, pdata['yval'], lamb)
	pyplot.plot(np.arange(1,m+1), error_train, np.arange(1,m+1), error_val)
	pyplot.title('Polynomial Regression Learning Curve (lambda = %s)'.format(lamb))
	pyplot.xlabel('Number of training examples')
	pyplot.ylabel('Error')
	#pyplot.legend('Train', 'Cross Validation')

	pyplot.figure(4)
	lamb_vec=np.array([0,0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
	(error_train, error_val)=validationCurve(lamb_vec, X_poly_no, pdata['y'], X_poly_val_no, pdata['yval'])
	pyplot.plot(lamb_vec, error_train, lamb_vec, error_val)

	lamb=3
	theta=trainLinearReg(X_poly_no, pdata['y'], lamb)
	error_test=linearRegCostFunction(X_poly_test_no, pdata['ytest'], theta, 0)
	print X_poly_no.shape
	print X_poly_val_no.shape
	print X_poly_test_no.shape
	print error_test
	pyplot.show()

if __name__=='__main__':
	main()	