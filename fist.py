import numpy as np
from dirs import generate_directions



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

def best_transform(X, Y):
    # a recommenter/recoder
    assert X.shape == Y.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t
            
def fist(X,a ):
    K = len(Xs)
    dim = Xs[0].shape[1]
    assert K == len(lambdas)
    X_tilde = np.zeros(X.shape)
    X_inv_hessian = np.eye(dim)*d
    dirs = generate_directions(3,100)

    # Newton's iterations
    for i in range(n_iter):
        X_grad = np.sum(np.stack([(X_tilde*dirs[i].reshape((1,-1))).sum(1) -
                                  (a[k](Xs[k])*dirs[i].reshape((1,-1))).sum(1)) / dirs.shape[0]),0)
        X_tilde = X_tilde - X_grad @ X_inv_hessian

        T,R,t = best_transform(X,X_tilde) # Transformation : Rotation + translation

        X_proj = T @ X_tilde
        yield X_proj
        
