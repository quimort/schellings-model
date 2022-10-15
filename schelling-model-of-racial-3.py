import numpy as np
from scipy.signal import convolve2d as convolve

# Gloval variables of the simulation
N = 7
sim_t = 0.4
empty = 0.1
A_to_B = 1
Kernel = np.array([[1,1,1],[1,0,1],[1,1,1]],dtype=np.int8)

def rand_init(N,empty,a_to_b):
    """ Random grid initialitzation
    A = 0
    B = 1
    empty = -1
    """
    vacant = N*N*empty
    population = N**2-vacant
    A = int(population*1/(1+1/a_to_b))
    B = int(population-A)
    M =np.zeros(int(N*N),dtype=np.int8)
    M[:B] = 1
    M[int(-vacant):] = -1
    np.random.shuffle(M)
    return  M.reshape(int(N),int(N))

def check_happines_neighborhod(M,position,boundary='wrap'):
    Kws = dict(mode='same',boundary=boundary)
    a_neights = convolve(M == 0,Kernel,**Kws)
    b_neights = convolve(M == 1,Kernel,**Kws)
    neights = convolve(M != -1,Kernel,**Kws)
    Kernel2 = np.array([[[1,-1],[1,0],[1,1]],[[0,-1],[0,0],[0,1]],[[-1,-1],[-1,0],[-1,1]]])
    salida = Kernel2 + position
    for ii in salida:
        dissatisfaction = 0
    

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
    dissatisfaction = a_dissatisfaction + b_dissatisfaction
    cordenates = np.where(dissatisfaction == True)
    index = np.vstack((cordenates[0], cordenates[1])).T
    random_number = np.random.randint(np.size(index,axis=0),size=1)
    random_index = index[random_number][0]
    cordenates_vacant = np.where((M == -1) == True)
    index_vacants = np.vstack((cordenates_vacant[0], cordenates_vacant[1])).T
    agent_tipe = M[random_index[0]][random_index[1]]
    for ii in index_vacants:
        if (agent_tipe == 0):
            position_a_neights = a_neights[ii[0]][ii[1]]
            position_neights = neights[ii[0]][ii[1]]
            if (position_a_neights/position_neights >= sim_t):
                M[random_index[0]][random_index[1]] = -1
                M[ii[0]][ii[1]] = 0
                break
        else:
            position_b_neights = b_neights[ii[0]][ii[1]]
            position_neights = neights[ii[0]][ii[1]]
            if (position_b_neights/position_neights >= sim_t):
                M[random_index[0]][random_index[1]] = -1
                M[ii[0]][ii[1]] = 0
                break

    return M
def get_mean_similarity_ratio(M,boundary='wrap'):

    Kws = dict(mode='same',boundary=boundary)
    a_neights = convolve(M == 0,Kernel,**Kws)
    b_neights = convolve(M == 1,Kernel,**Kws)
    neights = convolve(M != -1,Kernel,**Kws)
    n_similar_a = np.where(np.logical_and(M !=-1,M == 0),\
        a_neights/neights,0)
    n_similar_b = np.where(np.logical_and(M !=-1,M == 1),\
         b_neights/neights,0)
    n_similar = np.sum((n_similar_a+n_similar_b))
    return n_similar/np.size(M)

def get_mean_dissatisfaction(M,boundary='wrap'):

    Kws = dict(mode='same',boundary=boundary)
    a_neights = convolve(M == 0,Kernel,**Kws)
    b_neights = convolve(M == 1,Kernel,**Kws)
    neights = convolve(M != -1,Kernel,**Kws)
    a_dissatisfaction = (a_neights/neights < sim_t)&(M == 0)
    b_dissatisfaction = (b_neights/neights < sim_t)&(M == 1)

    n_a_dissatisfied, n_b_dissatisfied = a_dissatisfaction.sum(),b_dissatisfaction.sum()

    return (n_a_dissatisfied+n_b_dissatisfied)/np.size(M)

M = rand_init(N,empty,A_to_B)
posico = np.array([0,3])
check_happines_neighborhod(M,posico)
"""
similarity = get_mean_similarity_ratio(M)
dissatisfacton = get_mean_dissatisfaction(M)
print("similarity initial = {} /dissatisfaction initial = {}".format(similarity,dissatisfacton))
for i in range(5000):
    M = evolve(M)

similarity = get_mean_similarity_ratio(M)
dissatisfacton = get_mean_dissatisfaction(M)
print("similarity final = {} /dissatisfaction final = {}".format(similarity,dissatisfacton))
"""

