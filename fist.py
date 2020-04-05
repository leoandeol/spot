import numpy as np

def a(b):
    b[1]=4


def nn(X,Y):
    return ((X[:, :, None] - Y[:, :, None].T) ** 2).sum(1).argmin(1)



def assignment(X,Y,t):
    a = []
    m = X.shape[0]
    n = Y.shape[1]
    for mp in range(1,m):
        pass
        #if t[mp+1] >


def fist(Xs, lambdas,a ):
    K = len(Xs)
    assert K == len(lambdas)
    #Xt = np.
    

if __name__ == "__main__":
    b = [1,2,3]
    print(b)
    a(b)
    print(b)
