import numpy as np
import matplotlib.pyplot as plt
from dirs import generate_directions, Projector
from tqdm import tqdm

def cost(x, y):
    z = x-y
    return z*z
    
def sumCosts(X, Y, start1, start2, n):
    s = 0
    for i in range(n):
        s += cost(X[start1], Y[start2])
        start1 += 1
        start2 += 1
    return s

# Paper's implementation of the nn in 1-d
def nn_paper(X, Y, params, t):
    start0 = params["start0"]
    end0 = params["end0"]
    start1 = params["start1"]
    end1 = params["end1"]
    
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
        
def nn(X, Y):
    return ((X[:,None]-Y[:,None].T)**2).argmin(axis=1)

# Retrieves the right most index of the point from Y (in the range of the already assigned)
# that isn't assigned to a point from X'
def retrieve_s(a, start, end):
    s = a[end-1]
    """
    print(" ________ RETRIEVE S :")
    print("         start = ",start)
    print("         end = ",end)
    print("         a[end-1] = ",a[end-1])
    """
    for i in range(end-1,start-1,-1):
        checker = a[i]
        """
        print("         s = ",s)
        print("         checker = ",checker)
        """
        if checker != s:
            return s
        s -= 1
        
    return a[0] - 1
 
# Retrieves the index r of the point from X such that a[r] = s+1
# (s being the index given from the above function) 
def retrieve_r(a, s, start, end):
    for i in range(end-1,start-1,-1):
        if a[i] == s+1:
            return i
    return start
    
# Returns the sum of the costs when shifting all the current assignments to the left 
def sum_shifted_costs(X, Y, a, start, end):
    cost_ = 0
    for i in range(start,end):
        cost_ += cost(X[i],Y[a[i]-1])
    cost_ += cost(X[end],Y[a[end]])
    return cost_
    
# Returns the sum of the costs when keeping the current assignment and adding the new one on the right
def sum_non_shifted_costs(X, Y, a, start, end):
    """
    print(" ________ SUM NON SHIFTED COSTS")
    print("         end = ",end)
    print("         a[end-1] = ",a[end-1])
    print("         Y.shape[0] = ",Y.shape[0])
    """
    if a[end-1] >= Y.shape[0]-1:
        return np.inf
    cost_ = 0    
    for i in range(start,end):
        cost_ += cost(X[i],Y[a[i]])
    cost_ += cost(X[end],Y[a[end]+1])
    return cost_
    

# handles the case where the first sequence starts before or ends after the second sequence, 
# or where the NN of the first (resp. last) elements of X are the first (resp. last) elements of Y
# returns 1 if X entirely consumed ; 0 otherwise
def reduce_range(X, Y, a, params, value):
    a = a.astype(int)
    start0 = params["start0"]
    end0 = params["end0"]
    start1 = params["start1"]
    end1 = params["end1"]
    
    p0 = params
    
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
            
    params["start0"] = min0
    params["start1"] = cursor1
    
    if end0 == start0:
        value += localchange
        return 1
        
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
        
    params["end0"] = max0+1
    params["end1"] = cursor1b+1
    
    if end0 == start0:
        value += localchange
        return 1
        
    value += localchange
    return 0
    
# handles trivial cases: M==N, M==N-1, M==1, or nearest neighbor map is injective
# return 1 = subproblem solved
# return 0 = problem not solved
def handle_simple_cases(X, Y, a, t, params, value):   
    a = a.astype(int)
    t = t.astype(int)
    start0 = params["start0"]
    end0 = params["end0"]
    start1 = params["start1"]
    end1 = params["end1"]
     
    m = end0 - start0
    n = end1 - start1
    if m == 0 :
        return 1
    if m == n :
        d = 0
        for i in range(0,m):
            a[start0 + i] = i + start1
            d += cost(X[start0 + i], Y[start1 + i])
        
        value += d
        return 1
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
        return 1
    if m == 1:
        a[start0] = t[start0]
        value += cost(X[start0], Y[t[start0]])
        return 1
    
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
    return 0

def linear_time_decomposition(X, Y, assNN, params, new_params):
    assNN = assNN.astype(int)
    start0 = params["start0"]
    end0 = params["end0"]
    start1 = params["start1"]
    end1 = params["end1"]
    
    if end0 - start0 < 20:
        return False
    
    n = end1 - start1
    taken = np.ones(n).astype(int) * -1
    ninj = np.zeros(n).astype(int)
    
    taken[assNN[start0] - start1] = start0
    ninj[assNN[start0] - start1] += 1
    
    prev_free = np.arange(n).astype(int)
    next_free = np.arange(n).astype(int)
    first_right = assNN[start0 + 1] - start1
    last_left = assNN[start0 + 1] - start1
    for i in range(start0+1,end0):
        ass = assNN[i]
        assOffset = ass - start1
        ninj[assOffset] += 1
        if(taken[assOffset] < 0):
            taken[assOffset] = i
            first_right = assOffset
            last_left = assOffset
        else:
            if ninj[assOffset] > 1:
                cur = last_left-1
                while(cur >= 0):
                    if taken[cur] < 0 or cur == 0:
                        taken[cur] = i
                        prev_free[assOffset] = cur
                        next_free[cur] = next_free[assOffset]
                        break
                    else:
                        if prev_free[cur] == cur:
                            cur -= 1
                            cur = prev_free[cur]
                last_left = np.max([0,cur])
            else:
                prev_free[assOffset] = prev_free[prev_free[assOffset]]
            if first_right < n - 1:
                first_right += 1
            taken[first_right] = i
            prev_free[first_right] = last_left
            next_free[assOffset] = first_right
            next_free[last_left] = first_right
            
    lastStart = start0
    for i in range(start1, end1):
        assOffset = i - start1
        maxival = taken[assOffset]
        
        if taken[assOffset] >= 0:
            cur_params = {
                "start0" : 0,
                "end0" : 0,
                "start1" : 0,
                "end1" : 0
            }
            
            if next_free[assOffset] == assOffset:
                cur_params["start1"] = start1 + assOffset
                cur_params["start0"] = lastStart
                lastStart += 1
                cur_params["end0"] = cur_params["start0"] + 1
                cur_params["end1"] = cur_params["start1"] + 1
                new_params.append(cur_params)
            else:
                right = next_free[assOffset]
                while right < n-1 and next_free[right] != right:
                    right = next_free[right]
                for j in range(assOffset,right):
                    maxival = np.max([maxival,taken[j]])
                cur_params["start0"] = lastStart
                cur_params["end0"] = maxival + 1
                lastStart = cur_params["end0"]
                cur_params["start1"] = start1 + assOffset
                cur_params["end1"] = start1 + right + 1
                new_params.append(cur_params)
                i = start1 + right
    
    return True
    
def simple_solve(X, Y, params, assignment, assNN, value):
    assignment = assignment.astype(int)
    assNN = assNN.astype(int)
    
    n = params["end1"] - params["start1"]
    taken = np.ones(n).astype(int) * -1
    ninj = np.zeros(n).astype(int)
    taken[assNN[params["start0"]] - params["start1"]] = params["start0"]
    ninj[assNN[params["start0"]] - params["start1"]] += 1
    
    prev_free = np.arange(n).astype(int)
    next_free = np.arange(n).astype(int)
    cost_dontMove = np.zeros(n)
    cost_moveLeft = np.zeros(n)
    
    ass0 = assNN[params["start0"]]
    lastok = True
    
    cost_dontMove[assNN[params["start0"]] - params["start1"]] = cost(X[params["start0"]], Y[ass0])
    cost_moveLeft[assNN[params["start0"]] - params["start1"]] = np.inf if ass0==0 else cost(X[params["start0"]],Y[ass0-1])
    
    first_right = assNN[params["start0"] + 1] - params["start1"]
    last_left = assNN[params["start0"] + 1] - params["start1"]
    for i in range(params["start0"] + 1, params["end0"]):
        ass = assNN[i]
        assOffset = ass - params["start1"]
        ninj[assOffset] += 1
        if taken[assOffset] < 0:
            taken[assOffset] = i
            first_right = assOffset
            last_left = assOffset
            cost_dontMove[assOffset] = cost(X[i], Y[ass])
            cost_moveLeft[assOffset] = np.inf if ass==0 else cost(X[i], Y[ass-1])
        else:
            sumDontMove = 0
            sumMoveLeft = 0
            cur = prev_free[first_right] - 1
            isok = True
            while cur >= 0:
                sumDontMove += cost_dontMove[cur+1]
                if cost_moveLeft[cur+1] < 0:
                    isok = False
                sumMoveLeft += cost_moveLeft[cur+1]
                if taken[cur] < 0:
                    break
                else:
                    if prev_free[cur] == cur:
                        cur -= 1
                    else:
                        cur = prev_free[cur]-1
                        
            cdM = 0 
            cmL = 0
            
            if first_right >= n - 1:
                cdM = np.inf
            else:
                cdM = sumDontMove + cost(X[i], Y[params["start1"] + first_right + 1])
            if cur < 0:
                cmL = np.inf
            else:
                if isok:
                    cmL = sumMoveLeft + cost(X[i], Y[params["start1"] + first_right])
                else:
                    cmL = 0
                    if cur >= 0 and first_right < n - 1:
                        cmL = sumCosts(X,Y,(i-first_right-cur),(params["start1"]+cur),first_right-cur+1)
            
            if cmL < cdM or first_right >= n-1:
                last_left = np.max([0,cur])
                taken[last_left] = i 
                prev_free[assOffset] = prev_free[last_left]
                prev_free[first_right] = prev_free[last_left]
                prev_free[last_left] = next_free[first_right]
                lastok = False
                cost_dontMove[last_left] = cmL
                cost_moveLeft[last_left] = -1
            else:
                first_right += 1
                taken[first_right] = i
                prev_free[first_right] = prev_free[cur+1]
                prev_free[assOffset] = prev_free[cur+1]
                next_free[assOffset] = next_free[first_right]
                next_free[cur+1] = next_free[first_right]
                cost_dontMove[cur+1] = cdM
                cost_moveLeft[cur+1] = cmL
                lstok = True
    
    lastStart = params["start0"]
    for i in range(params["start1"], params["end1"]):
        assOffset = i - params["start1"]
        maxival = taken[assOffset]
        if taken[assOffset] >= 0:
            cur_params = {
                "start0" : 0,
                "end0" : 0,
                "start1" : 0,
                "end1" : 0
            }
            if next_free[assOffset] == assOffset:
                cur_params["start1"] = params["start1"] + assOffset
                cur_params["start0"] = lastStart
                lastStart += 1
                cur_params["end0"] = cur_params["start0"] + 1
                cur_params["end1"] = cur_params["start1"] + 1
                for j in range(cur_params["end0"] - cur_params["start0"]):
                    assignment[cur_params["start0"] + j] = cur_params["start1"] + j
                    value += cost(X[cur_params["start0"] + j], Y[cur_params["start1"] + j])
            else:
                right = next_free[assOffset]
                while right < n-1 and next_free[right] != right:
                    right = next_free[right]
                for j in range(assOffset,right):
                    maxival = np.max([maxival,taken[j]])
                cur_params["start0"] = lastStart
                cur_params["end0"] = maxival + 1
                lastStart = cur_params["end0"]
                cur_params["start1"] = params["start1"] + assOffset
                cur_params["end1"] = params["start1"] + right + 1
                for j in range(cur_params["end0"] - cur_params["start0"]):
                    assignment[cur_params["start0"] + j] = cur_params["start1"] + j
                    value += cost(X[cur_params["start0"] + j], Y[cur_params["start1"] + j])
                i = params["start1"] + right

def transport1d(X, Y, m0, n0, assignment, decomposition):
    assignment = assignment.astype(int)
    init_params = {
        "start0" : 0,
        "end0" : m0,
        "start1" : 0,
        "end1" : n0
    }
    value = 0
    assNN = np.zeros(m0).astype(int)
    nn_paper(X, Y, init_params,assNN)
    
    ret1 = reduce_range(X, Y, assignment, init_params, value)
    if ret1 == 1:
        return value

    nn_paper(X, Y, init_params, assNN)
    
    if decomposition:
        splits = []
        res = linear_time_decomposition(X, Y, assNN, init_params, splits)
        
        todo = []
        if res:
            for i in range(len(splits)):
                if splits[i]["end0"] == splits[i]["start0"]+1: 
                    # we directly handle problems of size 1 here
                    assignment[splits[i]["start0"]] = assNN[splits[i]["start0"]]
                    value += cost(X[splits[i]["start0"]], Y[assNN[splits[i]["start0"]]])
                else:
                    todo.append(splits[i])
        else:
            todo.append(init_params)
            
        for i in range(len(todo)):
            p = todo[i]
            nn_paper(X, Y, p, assNN)
            ret = handle_simple_cases(X, Y, assignment, assNN, p, value)
            if ret==1: 
                continue
            ret = reduce_range(X, Y, assignment, p, value)
            if ret==1: 
                continue
            ret = handle_simple_cases(X, Y, assignment, assNN, p, value)
            nn_paper(X, Y, p, assNN)
            simple_solve(X, Y, p, assignment, assNN, value)
    else:
        handle_simple_cases(X, Y, assignment, assNN, init_params, value)
        reduce_range(X, Y, assignment, init_params, value)
        handle_simple_cases(X, Y, assignment, assNN, init_params, value)
        nn_paper(X, Y, init_params, assNN)
        simple_solve(X, Y, init_params, assignment, assNN, value)
    
    return value

def correspondencesNd(cloud1, cloud2, n_iter, advect = False):
    assert cloud1.shape[1] == cloud2.shape[1]
    
    dim = cloud1.shape[1]
    m = cloud1.shape[0]
    n = cloud2.shape[0]
    cloud1Idx = []
    cloud2Idx = []
    projHist1 = -1*np.ones(m)
    projHist2 = -1*np.ones(n)
    ass1d = np.zeros(m).astype(int)
    d = 0
    
    for iter in range(n_iter):
        dir = generate_directions(dim,1)[0]
        proj = Projector(dir)
        
        for i in range(m):
            cloud1Idx.append((proj.proj(cloud1[i]), i))           
        for i in range(n):
            cloud2Idx.append((proj.proj(cloud2[i]), i))
            
        # can thread those sort
        cloud1Idx.sort()
        cloud2Idx.sort()

        for i in range(m):
            projHist1[i] = cloud1Idx[i][0]        
        for i in range(n):
            projHist2[i] = cloud2Idx[i][0]
            
        emd = transport1d(projHist1, projHist2, m, n, ass1d, False)
        d += emd
        
        if(advect):
            for i in range(m):
                for j in range(dim):
                    cloud1[cloud1Idx[i][1]][j] += (projHist2[ass1d[i]] - projHist1[i]) * dir[j]
        
    return 2*d/n_iter
    
def paper_fist(n_iter, n_slice, source_cloud, dest_cloud, rot_mat, trans_mat, useScaling, scaling):
    assert source_cloud.shape[1] == dest_cloud.shape[1]
    
    dim = source_cloud.shape[1]
    m = source_cloud.shape[0]
    n = dest_cloud.shape[0]
    
    scaling = 1
    rot_mat = np.identity(dim)
    trans_mat = np.zeros(dim)
    
    for iter in range(n_iter):
        source_copy = np.copy(source_cloud)
        print("doing correspondencesNd ...")
        correspondencesNd(source_copy, dest_cloud, n_slice, True)
        print("done !")
        
        center1 = np.zeros(dim)
        center2 = np.zeros(dim)
        for i in range(m):
            center1 += source_cloud[i]
            center2 += source_copy[i]
        center1 /= m
        center2 /= m
        
        cov = np.zeros((dim,dim))
        for i in range(m):
            p = source_cloud[i] - center1
            q = source_copy[i] - center2
            for j in range(dim):
                for k in range(dim):
                    cov[j,k] += q[j]*p[k]
        
        mat = np.copy(cov)
        U, S, V = np.linalg.svd(mat)
        orth = np.dot(U,V.T)
        d = np.linalg.det(orth)
        diag = np.identity(dim)
        diag[dim-1,dim-1] = d
        scal = 1
        
        if useScaling:
            std = 0
            for i in range(m):
                std += np.linalg.norm(source_cloud[i]-center1)
            s2 = 0
            for i in range(dim):
                s2 += np.absolute(S[i])
            scal = s2/std
            scaling *= scal
            
        rotM = np.dot(np.dot(U,diag),V.T)
        
        rot_mat = np.dot(rotM,rot_mat)
        trans_mat = trans_mat + center2 - center1
    
        for i in range(m):
            source_cloud[i] = scal*(rotM.dot(source_cloud[i]-center1)) + center2
        
        plt.scatter(source_cloud[:,0],source_cloud[:,1])
        plt.scatter(dest_cloud[:,0],dest_cloud[:,1])
        plt.show() 
        
    if useScaling:
        trans_mat = scaling*(rot_mat.dot(trans_mat))
    else:
        transG = rot_mat.dot(trans_mat)
    
def assignment_(X, Y):
    X.sort()
    Y.sort()
    m = X.shape[0]
    n = Y.shape[0]
    params = {
        "start0" : 0,
        "end0" : m,
        "start1" : 0,
        "end1" : n
    }
    
    a = np.zeros(m).astype(int)    
    t = np.zeros(m).astype(int)
    nn_paper(X, Y, params, t) # Nearest neighbor match between X and Y
    
    """
    plt.scatter(X,[1]*len(X))
    plt.scatter(Y,[0]*len(Y))
    for i in range(len(X)):
        plt.plot([X[i], Y[t[i]]], [1,0])
    plt.show()
    """
    
    cost_ = 0
    ret = reduce_range(X, Y, a, params, cost_)
    if ret == 1:
        return a
        
    nn_paper(X, Y, params, t) # Nearest neighbor match between X and Y
    #t = nn(X,Y) 
        
    ret = handle_simple_cases(X, Y, a, t, params, cost_)
    if ret == 1:
        return a  
    
    nn_paper(X, Y, params, t) # Nearest neighbor match between X and Y
    """
    print("Latest NN : ", t)
    print("Current assignment : ",a)
    print("Start 0 : ",start0)
    """     
    a[params["start0"]] = t[params["start0"]]
    for mp in range(params["start0"]+1,params["end0"]):
        if t[mp] > a[mp-1]:
                a[mp] = t[mp]            
        else:
            s = retrieve_s(a,0,mp)
            r = retrieve_r(a,s,0,mp)
            w1 = sum_shifted_costs(X,Y,a,r,mp)
            w2 = sum_non_shifted_costs(X,Y,a,r,mp)
            """            
            print("     s : ",s)
            print("     r : ",r)
            print("     w1 : ",w1)
            print("     w2 : ",w2)
            print("     mp : ",mp)
            print("     a[mp] :",a[mp])
            print("     a[mp-1] :",a[mp-1])
            print("     a[r:mp] :",a[r:mp])
            print("     a[r:mp-1] :",a[r:mp-1])
            print("     np.arange(s,a[mp]) :",np.arange(s,a[mp]))
            print("     np.arange(s,a[mp-1]) :",np.arange(s,a[mp-1]))
            """               
            if w1 < w2:
                # case 1
                a[mp] = a[mp-1]
                a[r:mp] = np.arange(s,a[mp-1])
            else:
                # case 2
                a[mp] = a[mp-1]+1
        #print("       Updated assignment : ",a)
        """
        plt.scatter(X,[1]*len(X))
        plt.scatter(Y,[0]*len(Y))
        for i in range(mp):
            plt.plot([X[i], Y[a[i]]], [1,0])
        plt.show()
        """
    
    while np.max(a) >= Y.shape[0]-1:
        for i in range(m):
            if(a[i] >= Y.shape[0]-1):
                val = a[i]
                s = retrieve_s(a,0,i)
                r = retrieve_r(a,s,0,i)
                """
                print("--- BEFORE ---")
                print("a : ",a)
                print("s : ",s)
                print("r : ",r)
                """
                a[i] = a[i-1]
                a[r:i] = np.arange(s,a[i-1])
                """
                print("--- AFTER ---")
                print("a : ",a)
                """
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

    ## special reflection case
    #if np.linalg.det(R) < 0:
    #   Vt[m-1,:] *= -1
    #   R = np.dot(Vt.T, U.T)

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

    for i in tqdm(range(n_iter)):
        #Find assignment
        a = []
        #print("iter",i)
        
        for j in range(n_dirs):
            X_proj = (X_tilde*dirs[j].reshape((1,-1))).sum(1)
            Y_proj = (Y*dirs[j].reshape((1,-1))).sum(1)
            a.append(assignment_(X_proj,Y_proj))
            """
            if 40 in a[-1]:
                print("error")
                plt.scatter(X_proj,[1]*len(X_proj))
                plt.scatter(Y_proj,[0]*len(Y_proj))
                for i in range(len(X_proj)):
                    if a[-1][i] != 40:
                        plt.plot([X_proj[i], Y_proj[a[-1][i]]], [1,0])
                plt.show()
            """
        #Newton's iteration
        X_grad = np.sum(np.stack([(X_tilde*dirs[k].reshape((1,-1))) -
                                  (Y[a[k]]*dirs[k].reshape((1,-1))) for k in range(n_dirs)]),0) / n_dirs
        #print(X_tilde.shape,"-",X_grad.shape,"@",X_inv_hessian.shape)
        X_tilde = X_tilde - X_grad @ X_inv_hessian
        #print(np.min(X_grad),np.max(X_grad))

        T,R,t = best_transform(X,X_tilde)
        # Transformation : Rotation + translation
        #print(X_tilde.shape,"@",R.shape,"+",t.shape)
        X_tilde = (X @ R ) + t.reshape((1,-1))
        #C = np.ones((X.shape[0], X.shape[1]+1))
        #C[:,:X.shape[1]] = np.copy(X)

        # Transform C
        #X_tilde = np.dot(T, C.T).T[:,:X.shape[1]]
        #print(X_tilde.shape, X.shape)
        yield X_tilde
        
