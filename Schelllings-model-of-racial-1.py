import numpy as np
from scipy.signal import convolve2d as convolve
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# Gloval variables of the simulation
N = 60
sim_t = 0.4
empty = 0.1
A_to_B = 1
Kernel = np.array([[1,1,1],[1,0,1],[1,1,1]],dtype=np.int8)

def rand_init(N,A_to_B,empty):
    """ Random grid initialitzation
    A = 1
    B = -1
    empty = 0
    """
    vacant = N*N*empty
    population = N*N-vacant
    A = int(population*1/(1+1/A_to_B))
    B = int(population-A)
    M =np.zeros(N*N,dtype=np.int8)
    M[:B] = -1
    M[-A:] = 1
    np.random.shuffle(M)
    return M.reshape(N,N)

def evolve(M,boundary='wrap'):
    """
    Args:
        M(numpy.array): the matrix to be evolved
        boundary(str): Either wrap, fill or symm
    if the siilarity ratio of neighbours
    to the enitre neghtbourhood polupation
    is lower than sim_t,
    then the individual moves to an empty house. 
    """
    Kws = dict(mode='same',boundary=boundary)
    a_neights = convolve(M == 1,Kernel,**Kws)
    b_neights = convolve(M == -1,Kernel,**Kws)
    neights = convolve(M != 0,Kernel,**Kws)

    
def showgrid(grid):

    values = np.unique(grid.ravel())
    labels = ['B','empty','A']
    plt.figure()
    im = plt.imshow(grid)
    colors = [ im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color 
    patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=labels[i]) ) for i in range(len(values)) ]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    plt.show()

grid = rand_init(N,A_to_B,empty)
showgrid(grid)


