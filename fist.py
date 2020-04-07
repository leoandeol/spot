import numpy as np
from dirs import generate_directions

def cost(x,y):
    z = x-y
    return z*z

# Paper's implementation of the nn in 1-d
def nn_paper(X,Y, start0, end0, start1, end1, t):
    cursor = start1
    for i in range(start0, end0):
        mind = np.inf
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
        t[i] = minj;

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
    costt = 0
    for i in range(start,end):
        costt += cost(X[i],Y[a[i]-1]) + cost(X[end+1],Y[a[end]])
    return costt
    
# Returns the sum of the costs when keeping the current assignment and adding the new one on the right
def sum_non_shifted_costs(X,Y,a,start,end):
    costt = 0    
    for i in range(start,end):
        costt += cost(X[i],Y[a[i]]) + cost(X[end+1],Y[a[end]+1])
    return costt

def assignment(X,Y):
    m = X.shape[0]
    n = Y.shape[0]
    a = np.zeros(m).astype(int)
    t = np.zeros(m).astype(int)
    nn_paper(X,Y, 0, m, 0, n, t) # Nearest neighbor match between X and Y
    a[0] = t[0]
    for mp in range(m):
        if t[mp+1] > a[mp]:
            a[mp+1] = t[mp+1]
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
    A = X
    B = Y

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
            
def fist(X,Y, n_iter, n_dirs):
    dim = X.shape[1]
    X_tilde = np.zeros(X.shape)
    X_inv_hessian = np.eye(dim)*dim
    dirs = generate_directions(dim,n_dirs)

    for i in range(n_iter):
        #Find assignment
        a = []
        for _ in range(n_dirs):
            a.append(assignment((X*dirs[i].reshape((1,-1))).sum(1),(Y*dirs[i].reshape((1,-1))).sum(1)))

        #Newton's iteration
        X_grad = np.sum(np.stack([(X_tilde*dirs[i].reshape((1,-1))).sum(1) -
                                  (X[a[i]]*dirs[i].reshape((1,-1))).sum(1) for i in range(n_dirs)] / n_dirs),0)
        X_tilde = X_tilde - X_grad @ X_inv_hessian

        T,R,t = best_transform(X,X_tilde) # Transformation : Rotation + translation

        X_proj = T @ X_tilde
        yield X_proj
        
