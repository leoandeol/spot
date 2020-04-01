




def nn(X,Y):
    return ((X[:, :, None] - Y[:, :, None].T) ** 2).sum(1).argmin(1)



def assignment(X,Y,t):
    a = []
    m = X.shape[0]
    n = Y.shape[1]
    for mp in range(1,m):
        if t[mp+1] > 

def __main__():
    
