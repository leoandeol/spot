import numpy as np
import argparse
from fist import fist
from data import data
from tqdm import tqdm
from time import time

parser = argparse.ArgumentParser(prog="FIST",description="Fast Iterative Sliced Transport (Bonneel et al, 2019) - Implemented by Leo Andeol (ENS Paris-Saclay) and Lothair Kizardjian (University Paris-Dauphine)")
parser.add_argument("--source", type=str)
parser.add_argument("--target", type=str)
parser.add_argument("--output", default="out/"+str(time)+".ply", type=str)
parser.add_argument("--iterations", default=5, type=int, help="Number of Iterations")
parser.add_argument("--directions", default=100, type=int, help="Number of randomly sampled directions")
#parser.add_argument('--cuda', dest='cuda', action='store_true')
#parser.add_argument('--no-cuda', dest='cuda', action='store_false')
#parser.set_defaults(cuda=torch.cuda.is_available())
#parser.add_argument('--verbose', dest='verbose', action='store_true')
#parser.set_defaults(verbose=False)
#parser.add_argument('--download', dest='download', action='store_true')
#parser.set_defaults(download=False)

args = parser.parse_args()

print(args.source)

