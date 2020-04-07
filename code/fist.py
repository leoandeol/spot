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

# Retrieves the right most index of the point from Y (in the range of the already assigned)
# that isn't assigned to a point from X'
def retrieve_s(a,start,end):
    s = a[end]
    for i in range(end,start,-1):
        checker = a[i]
        if checker != s:
            return s
        s -= 1
    return a[1] - 1
 
# Retrieves the index r of the point from X such that a[r] = s+1
# (s being the index given from the above function) 
def retrieve_r(a,s,start,end):
    for i in range(end,start,-1):
        if a[i] == s+1:
            return i
    # dont really know what to return if there is no match
    return -1
    
# Returns the sum of the costs when shifting all the current assignments to the left 
def sum_shifted_costs(X,Y,a,start,end):
    cost = 0
    for i in range(start,end):
        cost += cost(X[i],Y[a[i]-1]) + cost(X[end+1],Y[a[end]])
    return cost
    
# Returns the sum of the costs when keeping the current assignment and adding the new one on the right
def sum_non_shifted_costs(X,Y,a,start,end):
    cost = 0    
    for i in range(start,end):
        cost += cost(X[i],Y[a[i]]) + cost(X[end+1],Y[a[end]+1])
    return cost

def assignment(X,Y,t):
    m = X.shape[0]
    n = Y.shape[1]
    a = np.zeros(m)
    a[0] = t[0]
    for mp in range(m):
        if t[mp+1] > a[mp]:
            a[mp+1] = t[mp+1])
        else:
            s = retrieve_s(a,0,mp)
            r = retrieve_r(a,s,0,mp)
            w1 = sum_shifted_costs(X,Y,a,r,mp)
            w2 = sum_non_shifted_costs(X,Y,a,r,mp)
            if w1 < w2:
                # case 1
                a[mp+1] = a[mp]+1
                a[r:mp] = range(s,a[mp]-1)
            else:
                # case 2
                a[mp+1] = a[mp]+1
    return a

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
        
