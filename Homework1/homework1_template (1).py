from re import A
import numpy as np

def problem_1a (A, B):
    return A + B

def problem_1b (A, B, C):
    return (np.dot(A,B)-C)

def problem_1c (A, B, C):
    return (A*B)+(np.transpose(C))

def problem_1d (x, y):
    return np.inner(x,y)

def problem_1e (A, x):
    return (np.linalg.solve(A,x))

def problem_1f (A, x):
    At=np.transpose(A)
    xt=np.transpose(x)
    return (np.transpose(np.linalg.solve(At,xt)))

def problem_1g (A, i):
    return (np.sum(A[i-1, 1::2]))


def problem_1h (A, c, d):
    B=A[np.nonzero(A)]
    A_mean=np.mean([i for i in B if i>=c and i<=d])
    return A_mean 

def problem_1i (A, k):

    a,b = np.linalg.eig(A)
    print(a.shape)
    print(b.shape)
    idx_a = np.argsort(a)[::-1]
    i = idx_a[:k] 
    ans = b[:,i]
    return ans

def problem_1j (x, k, m, s):
    z= np.ones(len(x))
    #print(z.shape)
    mean=(x+m*z)
    cov = s*np.identity(len(x))
    l = np.random.multivariate_normal(mean, cov, k ).T
    return l

def problem_1k (A):
    return np.random.permutation(A)

def problem_1l (x):
    y=(x-np.mean(x))/np.std(x)
    return y

def problem_1m (x, k):
    #print(x.shape)
    j = x[:, np.newaxis]
    m= np.repeat(j,k,axis=1)
    return m

def problem_1n (X):
    x2=np.atleast_3d(X)
    x3=np.repeat(x2,X.shape[1],2)

    print(x3.shape)

    x5=X[np.newaxis,...]
    x6=np.repeat(x5,X.shape[1],0)
    x7=x6.copy()
    x8=np.swapaxes(x7,0,1)

    print(x8.shape)

    x9=np.square(x3-x8)
    print(x9.shape)
    D=np.sqrt(np.sum(x9,axis=0))

    return D

def linear_regression (X, y_tr):
    w=np.linalg.solve(np.dot(X,X.T),np.dot(X,y_tr))
    return w

def train_age_regressor ():
    # Load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
    ytr = np.load("age_regression_ytr.npy")
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    yte = np.load("age_regression_yte.npy")
    
    print("\n========== Given Data===========\n")
    X=X_tr.T
    # w is the weights vector with w1, to w2304 values 
    w = linear_regression(X, ytr)
    print("\n========== Calculated weights===========\n")
    print("W",w[1:10])
    print("shape of w ", w.shape)
    # y_hat is a predicted value from linear regression
    y_hat=np.dot(X_tr,w)
    loss=y_hat-ytr
    print("\n========== Predictions on train data===========\n")
    print("Predicted values - Ground truth = Error")
    list1=[print(" ",round(y_hat[i],1)," - ",ytr[i],"  =  ",round(loss[i],1))  for i in range (10)]
    #column vector with 5000 values
    F_MSE_train=np.square(y_hat-ytr)
    print("\nF_MSE_train first 10 values: \n",F_MSE_train[0:10])
    print("\nfinal value of F_MSE_train: \n",(np.mean(F_MSE_train)/2))
    print("===================For Test data==============")
    # y_hat_te is a predicted value from linear regression
    y_hat_te=np.dot(X_te,w)
    loss_te=y_hat_te-yte
    print("\n========== Predictions on test data===========\n")
    print("Predicted values - Ground truth = Error")
    list1=[print(" ",round(y_hat_te[i],1)," - ",yte[i],"  =  ",round(loss_te[i],1))  for i in range (10)]
    F_MSE_test=np.square(y_hat_te-yte)
    print("\nF_MSE_test first 10 values: \n",F_MSE_test[0:10])
    print("\nfinal value of F_MSE_test: \n",(np.mean(F_MSE_test)/2))


train_age_regressor()


x = np.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])

# print(problem_1n (x))

# print(problem_1h (x,2,9))