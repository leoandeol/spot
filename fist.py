import numpy as np



def cost(x,y):
    z = x-y
    return z*z

# Paper's implementation of the nn in 1-d
def nn_paper(X,Y, start0, end0, start1, end1, assignment):
    cursor = start1
    for i in range(start0, end0):
        mind = numpy.inf
        minj = -1
        for j in range(max(start1,cursor), end1):
            d = cost(X[i],Y[i])
            cursor = j-1
            if(d <= mind):
                mind = d
                minj = j
            else:
                if(d > mind+(1e-7)):
                    break;
        assignment[i] = minj;

def assignment(X,Y,t):
    a = []
    m = X.shape[0]
    n = Y.shape[1]
    a.append(t[0])
    for mp in range(1,m):
        if t[mp+1] > a[mp]:
            a.append(t[mp+1])
        else:


def fist(Xs, lambdas,a ):
    K = len(Xs)
    assert K == len(lambdas)
    #Xt = np.
    

if __name__ == "__main__":
    b = [1,2,3]
    print(b)
    a(b)
    print(b)
