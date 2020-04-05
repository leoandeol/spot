import numpy as np   
from math import sqrt, log, sin, cos, pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def BoxMuller():
    r = np.random.rand(2)
    r1, r2 = r[0], r[1]
    p = np.zeros((2))
    f = sqrt(-2 * log(max(1E-12, min(1. - 1E-12, r1))))
    p[0] = f * cos(2 * pi*r2)
    p[1] = f * sin(2 * pi*r2)
    return p;

DIM = 3
slices = 1000
dirs = np.zeros((slices, DIM))
n = 0
for slice in range(slices):
    for i in range(0,DIM,2):
        randGauss = BoxMuller()
        randGauss = np.random.randn(2)
        dirs[slice][i] = randGauss[0]
        n += dirs[slice][i] * dirs[slice][i]
        if i < DIM-1:
            dirs[slice][i + 1] = randGauss[1]
            n += dirs[slice][i + 1] * dirs[slice][i + 1]
    n = sqrt(n)
    for i in range(DIM):
        dirs[slice][i] /= n

        
    
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(dirs[:,0],dirs[:,1],dirs[:,2])
plt.show()
#for angle in range(0, 360):
#    ax.view_init(30, angle)
#    plt.draw()
#    plt.pause(.001)