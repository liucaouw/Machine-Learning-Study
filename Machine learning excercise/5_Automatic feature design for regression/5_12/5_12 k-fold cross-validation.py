from __future__ import division
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import pylab
import math

def read_data(filename):
	data = np.array(np.genfromtxt(filename, delimiter=','))
	X = np.reshape(data[:, 0], (1,np.size(data[:, 0])))
	y = np.reshape(data[:, 1], (np.size(data[:, 1]), 1))

	return X, y

def poly_basis(X, D):
	temp = np.ones((1, np.shape(X)[1]))
	F = X
	for i in range(2, D + 1):
		F = np.row_stack((F, X ** i))
	F = np.row_stack((temp, F))

	return F

def least_square_sol(F, y):
	'''
	Refer to eq. 5.19 in the text
	'''	
	w = np.dot(np.dot(np.linalg.pinv(np.dot(F, F.T)), F), y)

	return w

def mean_square_error(w, F, y):
    
	res = np.dot(F.T, w) - y
	res = res * res
	mse = np.mean(res, axis=0)

	return mse

def random_split(P, K):

	temp = np.random.permutation(P)
	folds = np.split(temp, K)

	return folds

def train_val_split(X, y, folds, fold_id):
    
	X_val = np.zeros((1, len(folds[fold_id])))
	y_val = np.zeros((len(folds[fold_id]), 1))
	X_train = np.zeros((1, (len(folds) - 1) * len(folds[fold_id])))
	y_train = np.zeros(((len(folds) - 1) * len(folds[fold_id]), 1))
	j = 0
	for i in folds[fold_id]:
		X_val[0][j] = X[0][i]
		y_val[j][0] = y[i][0]
		j = j + 1

	count = 0
	for k in range(0, len(folds)):
		if k != fold_id:
			for l in folds[k]:
				X_train[0][count] = X[0][l]
				y_train[count][0] = y[l][0]
				count = count + 1

	return X_train, y_train, X_val, y_val

def make_plot(D, MSE_train, MSE_val):
	plt.figure()
	train, = plt.plot(D, MSE_train, 'yv--')
	val, = plt.plot(D, MSE_val, 'bv--')
	plt.legend(handles=[train, val], labels=['training error', 'testing error'])
	plt.xlabel('Degree of polynomial basis')
	plt.ylabel('average error(in log scale)')
	plt.yscale('log')
	plt.show()

def load_data():
    data = np.array(np.genfromtxt('galileo_ramp_data.csv', delimiter=','))
    x = np.reshape(data[:,0],(np.size(data[:,0]),1))
    y = np.reshape(data[:,1],(np.size(data[:,1]),1))
    return x,y

# YOUR CODE GOES HERE takes poly features of the input 
def poly_features(x,D):
    F = []
    for i in x:
        F.append(1)
        for j in range(1,D+1):
            F.append(i**j)
    F=np.array(F)
    F.shape=(len(x),D+1)
    F=F.T
    return F

# plot the polynomial  
def plot_model(w,D):
    # plot determined surface in 3d space
    s = np.arange(0,10,.01)
    f = poly_features(s,D)
    z = np.dot(f.T,w)

    # plot contour in original space
    plt.plot(s,z, color = 'r', linewidth = 2)
    plt.ylim([-1,2])
    plt.xlim([0,10])

# plot data 
def plot_data(x,y,deg):
        plt.scatter(x,y,s = 30, color = 'k')
        
# run over all the degrees, fit each models, and calculate errors
def try_all_degs(x,y,deg_range):
    # plot datapoints - one panel for each deg in deg_range
    plot_data(x,y,deg_range)

    # generate nonlinear features
    mses = []

    for D in np.arange(0,np.size(deg_range)):
        # generate poly feature transformation
        F = poly_features(x,deg_range[D])

        # get weights for current model
        temp = np.linalg.pinv(np.dot(F,F.T))
        w = np.dot(np.dot(temp,F),y)
        MSE = np.linalg.norm(np.dot(F.T,w)-y)/np.size(y)
        mses.append(MSE)

        # plot fit to data
        plot_model(w,deg_range[D])
    
# load data and defined degree range
X, y = read_data('galileo_ramp_data.csv')
num_fold, num_degree = 6, 6
folds = random_split(P=y.size, K=num_fold)

MSE_train, MSE_val = [0]*num_degree, [0]*num_degree
D = np.arange(1, num_degree+1)
for f in range(num_fold):
	X_train, y_train, X_val, y_val = train_val_split(X, y, folds, fold_id=f)
	for i, d in enumerate(D):
		F_train = poly_basis(X_train, D=d)
		F_val = poly_basis(X_val, D=d)
		w = least_square_sol(F_train, y_train)
		MSE_train[i]=MSE_train[i]+mean_square_error(w, F_train, y_train)/num_fold
		MSE_val[i]=MSE_val[i]+ mean_square_error(w, F_val, y_val)/num_fold

print ('The best degree of polynomial basis is %d' % (MSE_val.index(min(MSE_val))+1))

make_plot(D, MSE_train, MSE_val)

x, y = load_data()
deg_range =[(MSE_val.index(min(MSE_val))+1)]          # degree polys to try

# run all over degree range
try_all_degs(x,y,deg_range)