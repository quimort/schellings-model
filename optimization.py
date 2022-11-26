import numpy as np
from scipy.signal import convolve2d as convolve
import time

N = 100
sim_t = 0.5
empty = 0.01
A_to_B = 1
Kernel = np.array([[1,1,1],[1,0,1],[1,1,1]],dtype=np.int8)
Kernel2 = np.array([[1,-1],[1,0],[1,1],[0,-1],[0,1],[-1,-1],[-1,0],[-1,1]])
epsilon = 0.00001
def rand_init(N,empty,a_to_b):
    """ Random grid initialitzation
    A = 0
    B = 1
    empty = -1
    """
    vacant = N*N*empty
    population = N*N-vacant
    A = int(population*1/(1+1/a_to_b))
    B = int(population-A)
    M =np.zeros(int(N*N),dtype=np.int8)
    M[:B] = 1
    M[int(-vacant):] = -1
    np.random.shuffle(M)
    return  M.reshape(int(N),int(N))

def convolve_method(M,boundary='wrap'):
    Kws = dict(mode='same',boundary=boundary)
    a_neights = convolve(M == 0,Kernel,**Kws)

    return M,a_neights

def normal_method(M,a_neights):
    for i in range(np.size(M,axis=0)):
        for j in range(np.size(M,axis=1)):
            neights_pos = [i,j]+Kernel2
            neights_pos = np.where(neights_pos == np.size(M,axis=0),0,neights_pos)
            Y = np.transpose(neights_pos)[0]
            X = np.transpose(neights_pos)[1]
            neights = M[Y,X]
            a_neight = (neights == 0).sum() 
            a_neights[i][j] = a_neight
    
    return M,a_neights

M = rand_init(N,empty,A_to_B)
start_time = time.time()
M,a_neights1 = convolve_method(M)
print("--- %s seconds --- optimization" % (time.time() - start_time))
start_time = time.time()
a_neights2=np.zeros((N,N),dtype=np.int8)
M,a_neights2 = normal_method(M,a_neights2)
print("--- %s seconds --- normal" % (time.time() - start_time))
print(np.allclose(a_neights1,a_neights2))
