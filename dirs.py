import numpy as np   
from math import sqrt, log, sin, cos, pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_directions(dim=3, slices=100):
    #dirs = np.zeros((slices, dim))
    #for slice in range(slices):
    #    n = 0
    #    for i in range(0,dim,2):
    #        randGauss = np.random.randn(2)
    #        dirs[slice][i] = randGauss[0]
    #        n += dirs[slice][i] * dirs[slice][i]
    #        if i < dim-1:
    #            dirs[slice][i + 1] = randGauss[1]
    #            n += dirs[slice][i + 1] * dirs[slice][i + 1]
    #    n = sqrt(n)
    #    for i in range(dim):
    #        dirs[slice][i] /= n
    dirs = np.random.randn(slices,dim)
    dirs /= np.linalg.norm(dirs,axis=1)[:,None]
    return dirs

if __name__ == "__main__":
    dirs = generate_directions()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dirs[:,0],dirs[:,1],dirs[:,2])
    plt.show()
    #for angle in range(0, 360):
    #    ax.view_init(30, angle)
    #    plt.draw()
    #    plt.pause(.001)
    
