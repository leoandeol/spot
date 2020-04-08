import numpy as np
import argparse
from fist import fist
from data import read_ply, write_ply
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(prog="FIST",description="Fast Iterative Sliced Transport (Bonneel et al, 2019) - Implemented by Leo Andeol (ENS Paris-Saclay) and Lothair Kizardjian (University Paris-Dauphine)")
parser.add_argument("--source", type=str)
parser.add_argument("--target", type=str)
parser.add_argument("--output", default="out/"+str(time)+".ply", type=str)
parser.add_argument("--iterations", default=20, type=int, help="Number of Iterations")
parser.add_argument("--directions", default=100, type=int, help="Number of randomly sampled directions")
parser.add_argument('--demo', dest='demo', action='store_true')
#parser.add_argument('--no-cuda', dest='cuda', action='store_false')
#parser.set_defaults(cuda=torch.cuda.is_available())

args = parser.parse_args()

if args.demo:
    nx = 20
    ny = 40
    tx = np.linspace(0,np.pi,nx)+1e-2
    ty = np.linspace(0,2*np.pi,ny)
    x = np.zeros((nx,2))
    x[:,0] = np.cos(tx)/2-.33
    x[:,1] = np.sin(tx)/2+1
    x = x @ np.array([[np.cos(40),-np.sin(40)],[np.sin(40),np.cos(40)]])
    y = np.zeros((ny,2))
    y[:,0] = np.cos(ty)
    y[:,1] = np.sin(ty)/1.5

    plt.scatter(x[:,0],x[:,1])
    plt.scatter(y[:,0],y[:,1])
    plt.show()

    x_projs = list(fist(x,y,args.iterations,args.directions))
    x_last_proj = x_projs[-1]

    plt.scatter(x_last_proj[:,0],x_last_proj[:,1])
    plt.scatter(y[:,0],y[:,1])
    plt.show()
    
else:
    pass

