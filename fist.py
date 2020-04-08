import numpy as np
import matplotlib.pyplot as plt
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
            d = cost(X[i],Y[j])
            cursor = j-1
            if(d <= mind):
                mind = d
                minj = j
            else:
                if(d > mind+1e-5):
                    break;
        t[i] = minj;
        
def nn(X,Y):
    return ((X[:,None]-Y[:,None].T)**2).argmin(axis=1)

# Retrieves the right most index of the point from Y (in the range of the already assigned)
# that isn't assigned to a point from X'
def retrieve_s(a,start,end):
    s = a[end]
    for i in range(end,start,-1):
        checker = a[i]
        if checker != s:
            return s
        s -= 1
    return a[0] - 1
 
# Retrieves the index r of the point from X such that a[r] = s+1
# (s being the index given from the above function) 
def retrieve_r(a,s,start,end):
    for i in range(end,start,-1):
        if a[i] == s+1:
            return i
    return start
    
# Returns the sum of the costs when shifting all the current assignments to the left 
def sum_shifted_costs(X,Y,a,start,end):
    if a[end] >= Y.shape[0]-1 or end == X.shape[0]-1:
        return np.inf
    cost_ = 0
    for i in range(start,end):
        cost_ += cost(X[i],Y[a[i]-1])
    cost_ += cost(X[end+1],Y[a[end]])
    return cost_
    
# Returns the sum of the costs when keeping the current assignment and adding the new one on the right
def sum_non_shifted_costs(X,Y,a,start,end):
    if a[end] >= Y.shape[0]-1 or end == X.shape[0]-1:
        return np.inf
    cost_ = 0    
    for i in range(start,end):
        cost_ += cost(X[i],Y[a[i]])
    cost_ += cost(X[end+1],Y[a[end]+1])
    return cost_
    

# handles the case where the first sequence starts before or ends after the second sequence, 
# or where the NN of the first (resp. last) elements of X are the first (resp. last) elements of Y
# returns 1 if X entirely consumed ; 0 otherwise
def reduce_range(X,Y,a,start0,end0,start1,end1,value):

    # X (partly) at the left of Y : can match the outside of X to the begining of Y
    cursor1 = start1
    min0 = start0
    localchange = 0
    for i in range(start0,end0):
        if X[i] <= Y[cursor1]:
            a[i] = cursor1
            localchange += cost(X[i],Y[cursor1])
            cursor1 += 1
            min0 = i+1
        else:
            break
            
    start0 = min0
    start1 = cursor1
    
    if end0 == start0:
        value += localchange
        return 1,start0,end0,start1,end1
        
    # X (partly) at the right of Y : can match the outside of X to the end of Y
    cursor1b = end1-1
    max0 = end0-1
    for i in range(end0-1,start0,-1):
        if X[i] >= Y[cursor1b]:
            a[i] = cursor1b
            localchange += cost(X[i],Y[cursor1b])
            cursor1b -= 1
            max0 = i-1
        else:
            break
        
    end0 = max0+1
    end1 = cursor1b+1
    
    if end0 == start0:
        value += localchange
        return 1,start0,end0,start1,end1
        
    value += localchange
    return 0,start0,end0,start1,end1
    
# handles trivial cases: M==N, M==N-1, M==1, or nearest neighbor map is injective
# return 1 = subproblem solved
# return 0 = problem not solved
def handle_simple_cases(X,Y,a,t,start0,end0,start1,end1,value):    
    m = end0 - start0
    n = end1 - start1
    if m == 0 :
        return 1,start0,end0,start1,end1
    if m == n :
        d = 0
        for i in range(0,m):
            a[start0 + i] = i + start1
            d += cost(X[start0 + i], Y[start1 + i])
        
        value += d
        return 1,start0,end0,start1,end1
    if m == n - 1:
        d1 = 0
        d2 = 0
        for i in range(0,m):
            d2 += cost(X[start0 + i], Y[start1 + i + 1])
        d = 0
        b = d2
        best_s = d2
        best_i = -1
        for i in range(0,m):
            d1 += cost(X[start0 + i], Y[start1 + i]) #forward cost
            b -= cost(X[start0 + i], Y[start1 + i + 1]) #backward cost
            s = b + d1
            if s < best_s:
                best_s = s
                best_i = i
        
        for i in range(0,m):
            if i < best_i:
                a[start0 + i] = i + start1
            else:
                a[start0 + i] = i + 1 + start1
        
        value += best_s
        return 1,start0,end0,start1,end1
    if m == 1:
        a[start0] = t[start0]
        value += cost(X[start0], Y[t[start0]])
        return 1,start0,end0,start1,end1
    
    """
    #checks if nn is injective
    curId = 0
    sumMin = 0
    valid = True
    for i in range(0,m):
        assignment = 0
        h1 = X[start0 + i]
        mini = np.inf
        for j in range(curId,n):
            v = cost(h1,Y[start1 + j])
            curId = j
            if v < mini:
                mini = v
                assignment = j + start1
            if j < n -1:
                vnext = cost(h1,Y[start1 + j + 1])
                if vnext > v :
                    break
        if mini == np.inf:
            valid = False
            break
        if i > 0 and assignment == a[start0 + i + 1]:
            valid = False
            break
        sumMin += mini
        a[start0 + i] = assignment
    if valid:
        value += sumMin
        return 1,start0,end0,start1,end1
    """
    return 0,start0,end0,start1,end1


def assignment(X,Y):
    X.sort()
    Y.sort()
    m = X.shape[0]
    n = Y.shape[0]
    start0 = 0
    end0 = m
    start1 = 0
    end1 = n
    
    a = np.zeros(m).astype(int)    
    t = np.zeros(m).astype(int)
    nn_paper(X,Y,start0,end0,start1,end1, t) # Nearest neighbor match between X and Y
    
    """
    plt.scatter(X,[1]*len(X))
    plt.scatter(Y,[0]*len(Y))
    for i in range(len(X)):
        plt.plot([X[i], Y[t[i]]], [1,0])
    plt.show()
    """
    
    cost_ = 0
    ret,start0,end0,start1,end1 = reduce_range(X,Y,a,start0,end0,start1,end1,cost_)
    if ret == 1:
        return a
        
    nn_paper(X,Y,start0,end0,start1,end1, t) # Nearest neighbor match between X and Y
    #t = nn(X,Y) 
        
    ret,start0,end0,start1,end1 = handle_simple_cases(X,Y,a,t,start0,end0,start1,end1,cost_)
    if ret == 1:
        return a  
    
    nn_paper(X,Y,start0,end0,start1,end1, t) # Nearest neighbor match between X and Y
            
    a[start0] = t[start0]
    for mp in range(start0+1,end0):
        if t[mp] > a[mp-1]:
                a[mp] = t[mp]            
        else:
            s = retrieve_s(a,start0,mp)
            r = retrieve_r(a,s,start0,mp)
            w1 = sum_shifted_costs(X,Y,a,r,mp)
            w2 = sum_non_shifted_costs(X,Y,a,r,mp)
            if w1 < w2:
                # case 1
                a[mp] = a[mp-1]
                a[r:mp] = np.arange(s,a[mp])
            else:
                # case 2
                a[mp] = a[mp-1]+1
        """
        plt.scatter(X,[1]*len(X))
        plt.scatter(Y,[0]*len(Y))
        for i in range(mp):
            plt.plot([X[i], Y[a[i]]], [1,0])
        plt.show()
        """
    while 40 in a:
        s = retrieve_s(a,start0,m-1)
        r = retrieve_r(a,s,start0,m-1)
        print("s : ",s)
        print("r : ",r)
        a[mp] = a[mp-1]
        a[r:mp] = np.arange(s,a[mp])

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
    X_tilde = np.copy(X)
    X_inv_hessian = np.eye(dim)*dim
    dirs = generate_directions(dim,n_dirs)

    for i in range(n_iter):
        #Find assignment
        a = []
        print("iter",i)
        
        for j in range(n_dirs):
            X_proj = (X*dirs[j].reshape((1,-1))).sum(1)
            Y_proj = (Y*dirs[j].reshape((1,-1))).sum(1)
            a.append(assignment(X_proj,Y_proj))
            if 40 in a[-1]:
                print(dirs[j])
            #if 40 in a[-1]:
            #    print("error")
            #    plt.scatter(X_proj,[1]*len(X_proj))
            #    plt.scatter(Y_proj,[0]*len(Y_proj))
            #    for i in range(len(X_proj)):
            #        if a[-1][i] != 40:
            #            plt.plot([X_proj[i], Y_proj[a[-1][i]]], [1,0])
            #    plt.show()
            
        #Newton's iteration
        X_grad = np.sum(np.stack([(X_tilde*dirs[k].reshape((1,-1))) -
                                  (Y[a[k]]*dirs[k].reshape((1,-1))) for k in range(n_dirs)]),0) / n_dirs
        #print(X_tilde.shape,"-",X_grad.shape,"@",X_inv_hessian.shape)
        X_tilde = X_tilde - X_grad @ X_inv_hessian
        #print(np.min(X_grad),np.max(X_grad))

        T,R,t = best_transform(X,X_tilde)
        # Transformation : Rotation + translation
        #print(X_tilde.shape,"@",R.shape,"+",t.shape)
        X_tilde = (X @ R) - t[None,:]
        #C = np.ones((X.shape[0], X.shape[1]+1))
        #C[:,:X.shape[1]] = np.copy(X)

        # Transform C
        #X_tilde = np.dot(T, C.T).T[:,:X.shape[1]]
        #print(X_tilde.shape, X.shape)
        yield X_tilde
        
