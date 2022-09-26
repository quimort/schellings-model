from operator import ne
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
    A = 0
    B = 1
    empty = -1
    """
    vacant = N*N*empty
    population = N*N-vacant
    A = int(population*1/(1+1/A_to_B))
    B = int(population-A)
    M =np.zeros(N*N,dtype=np.int8)
    M[:B] = 1
    M[int(-vacant):] = -1
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
    a_neights = convolve(M == 0,Kernel,**Kws)
    b_neights = convolve(M == 1,Kernel,**Kws)
    neights = convolve(M != -1,Kernel,**Kws)
    a_dissatisfaction = (a_neights/neights < sim_t)&(M == 0)
    b_dissatisfaction = (b_neights/neights < sim_t)&(M == 1)
    M[b_dissatisfaction | a_dissatisfaction] = -1
    vacant = (M == -1).sum()

    n_a_dissatisfied, n_b_dissatisfied = a_dissatisfaction.sum(),b_dissatisfaction.sum()
    filling = -np.ones(vacant,dtype=np.int8)
    filling[:int(n_a_dissatisfied)] = 0
    filling[int(n_b_dissatisfied):int(n_b_dissatisfied+n_a_dissatisfied)] = 1
    np.random.shuffle(filling)
    M[M == -1] = filling



    
def showgrid(grid):

    values = np.unique(grid.ravel())
    labels = ['empty','A','B']
    plt.figure()
    plt.subplot(2,2,1)
    plt.title("initial")
    im = plt.imshow(grid)
    colors = [ im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color 
    patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=labels[i]) ) for i in range(len(values)) ]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    counter = 0
    while(counter<30):
        evolve(grid)
        counter = counter + 1
    plt.subplot(2,2,2)
    plt.title("30 loops")
    im = plt.imshow(grid)
    colors = [ im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color 
    patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=labels[i]) ) for i in range(len(values)) ]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    while(counter<60):
        evolve(grid)
        counter = counter + 1
    plt.subplot(2,2,3)
    plt.title("60 loops")
    im = plt.imshow(grid)
    colors = [ im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color 
    patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=labels[i]) ) for i in range(len(values)) ]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    while(counter<120):
        evolve(grid)
        counter = counter + 1
    plt.subplot(2,2,4)
    plt.title("120 loops")
    im = plt.imshow(grid)
    colors = [ im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color 
    patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=labels[i]) ) for i in range(len(values)) ]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    plt.show()

grid = rand_init(N,A_to_B,empty)
showgrid(grid)



