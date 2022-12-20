from itertools import count
import numpy as np
from scipy.signal import convolve2d as convolve
import time
from multiprocessing import Pool
import os
# Gloval variables of the simulation
N = 30
sim_t = 0.5
empty = 0.1
A_to_B = 1
epsilon=0.00001
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
def evolve1(M,bloked,blocks_a,blocks_b,boundary='wrap'):
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
    a_dissatisfaction = (a_neights < sim_t*neights)&(M == 0)
    b_dissatisfaction = (b_neights < sim_t*neights)&(M == 1)
    n_a_dissatisfied, n_b_dissatisfied = a_dissatisfaction.sum(),b_dissatisfaction.sum()
    dissatisfaction_n = (n_a_dissatisfied+n_b_dissatisfied)
    cordenates_a = np.argwhere(a_dissatisfaction)
    cordenates_b = np.argwhere(b_dissatisfaction)
    cordenates = np.concatenate((cordenates_a,cordenates_b),axis = 0)
    if (np.size(cordenates,axis=0) == 0):
        bloked = True
        return M,dissatisfaction_n,bloked,blocks_a,blocks_b
    random_number = np.random.randint(np.size(cordenates,axis=0),size=1)
    random_index = cordenates[random_number][0]
    index_vacants = np.argwhere(M == -1)
    agent_tipe = M[random_index[0]][random_index[1]]
    Y = np.transpose(index_vacants)[0]
    X = np.transpose(index_vacants)[1]
    if (agent_tipe == 0 ):
        a_neights_vacants = a_neights[Y,X]
        neights_vacants = neights[Y,X]
        satisfaying_vacants_a = (a_neights_vacants >= sim_t*neights_vacants)
        if(True in satisfaying_vacants_a):
            array_of_good_vacants = np.where(satisfaying_vacants_a == True)
            move_to = index_vacants[array_of_good_vacants[0][0]]
            M[random_index[0]][random_index[1]] = -1
            M[move_to[0]][move_to[1]] = 0
        if( True not in satisfaying_vacants_a):
            blocks_a = True
    else:
        b_neights_vacants = b_neights[Y,X]
        neights_vacants = neights[Y,X]
        satisfaying_vacants_b = (b_neights_vacants >= sim_t*neights_vacants)
        if(True in satisfaying_vacants_b):
            array_of_good_vacants = np.where(satisfaying_vacants_b == True)
            move_to = index_vacants[array_of_good_vacants[0][0]]
            M[random_index[0]][random_index[1]] = -1
            M[move_to[0]][move_to[1]] = 1
        if( True not in satisfaying_vacants_b):
            blocks_b = True
    if(blocks_a == True):
        "a agents are blocked"
        index_test = cordenates_b
        if(np.size(index_test,axis=0) != 0):
            cordenate_test = index_test[0]
            b_neights_vacants = b_neights[Y,X]
            neights_vacants = neights[Y,X]
            satisfaying_vacants_b = (b_neights_vacants >= sim_t*neights_vacants)
            if(True in satisfaying_vacants_b):
                array_of_good_vacants = np.where(satisfaying_vacants_b == True)
                move_to = index_vacants[array_of_good_vacants[0][0]]
                M[cordenate_test[0]][cordenate_test[1]] = -1
                M[move_to[0]][move_to[1]] = 1
            if( True not in satisfaying_vacants_b):
                blocks_b = True
    if(blocks_b == True):
        "b agents are blocked"
        index_test = cordenates_a
        if(np.size(index_test,axis=0) != 0):
            cordenate_test = index_test[0]
            a_neights_vacants = a_neights[Y,X]
            neights_vacants = neights[Y,X]
            satisfaying_vacants_a = (a_neights_vacants >= sim_t*neights_vacants)
            if(True in satisfaying_vacants_a):
                array_of_good_vacants = np.where(satisfaying_vacants_a == True)
                move_to = index_vacants[array_of_good_vacants[0][0]]
                M[cordenate_test[0]][cordenate_test[1]] = -1
                M[move_to[0]][move_to[1]] = 1
            if( True not in satisfaying_vacants_a):
                blocks_a = True
    if(blocks_a == True and blocks_b == True):
       bloked= True
    
    return M,dissatisfaction_n,bloked,blocks_a,blocks_b

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
    if (np.size(index,axis=0) == 0):
        return M
    random_number = np.random.randint(np.size(index,axis=0),size=1)
    random_index = index[random_number][0]
    cordenates_vacant = np.where((M == -1) == True)
    index_vacants = np.vstack((cordenates_vacant[0], cordenates_vacant[1])).T
    agent_tipe = M[random_index[0]][random_index[1]]
    foundit = False
    counter = 0
    while (not (foundit == True)) and (counter < np.size(index_vacants,axis=0)):
        if (agent_tipe == 0):
            position_a_neights = a_neights[index_vacants[counter][0]][index_vacants[counter][1]]
            position_neights = neights[index_vacants[counter][0]][index_vacants[counter][1]]
            if (position_a_neights/position_neights >= sim_t):
                M[random_index[0]][random_index[1]] = -1
                M[index_vacants[counter][0]][index_vacants[counter][1]] = 0
                foundit = True
        else:
            position_b_neights = b_neights[index_vacants[counter][0]][index_vacants[counter][1]]
            position_neights = neights[index_vacants[counter][0]][index_vacants[counter][1]]
            if (position_b_neights/position_neights >= sim_t):
                M[random_index[0]][random_index[1]] = -1
                M[index_vacants[counter][0]][index_vacants[counter][1]] = 0
                foundit = True
        counter += 1
    """
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
    """
    return M
def start(empty):
    M = rand_init(N,empty,A_to_B)
    bloked = False
    blocks = np.array([False,False])
    counter = 0
    start_time = time.time()
    M,dissatisfaction_n,bloked,blocks = evolve1(M,bloked,blocks)
    time1 = time.time() - start_time
    start_time = time.time()
    M = evolve(M)
    time2 = time.time() - start_time
    return time1,time2

    


file_name = "schelling_values_optimization_model_2_30.csv"
emptines = np.logspace(-2,0,100)
f = open(file_name, "w")
f.write("empty;time opimaized;time no optimized")
for i in emptines:
    time1,time2 = start(i)
    f.write("{};{};{}".format(i,time1,time2))
f.close()  
        