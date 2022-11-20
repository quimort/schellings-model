from itertools import count
import numpy as np
from scipy.signal import convolve2d as convolve
import time
from multiprocessing import Pool
import os
# Gloval variables of the simulation
N = 50
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
def inicialize_empty(emptines):
    global empty

    empty = emptines
def evolve(M,bloked,blocks_a,blocks_b,boundary='wrap'):
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
    dissatisfaction_n = (n_a_dissatisfied+n_b_dissatisfied)/np.size(M)
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

def get_mean_similarity_ratio(M,boundary='wrap'):

    Kws = dict(mode='same',boundary=boundary)
    a_neights = convolve(M == 0,Kernel,**Kws)
    b_neights = convolve(M == 1,Kernel,**Kws)
    neights = convolve(M != -1,Kernel,**Kws)
    a_neights = a_neights + epsilon
    b_neights = b_neights + epsilon
    neights = neights + epsilon
    n_similar_a = np.where(np.logical_and(np.logical_and(M !=-1,M == 0),neights != 0),\
        a_neights/neights,0)
    n_similar_b = np.where(np.logical_and(np.logical_and(M !=-1,M == 1),neights != 0),\
         b_neights/neights,0)
    n_similar = np.sum((n_similar_a+n_similar_b))
    return n_similar/np.size(M)

def get_mean_dissatisfaction(M,boundary='wrap'):

    Kws = dict(mode='same',boundary=boundary)
    a_neights = convolve(M == 0,Kernel,**Kws)
    b_neights = convolve(M == 1,Kernel,**Kws)
    neights = convolve(M != -1,Kernel,**Kws)
    a_neights = a_neights + epsilon
    b_neights = b_neights + epsilon
    neights = neights + epsilon
    a_dissatisfaction = (a_neights/neights < sim_t)&(M == 0)
    b_dissatisfaction = (b_neights/neights < sim_t)&(M == 1)
    n_a_dissatisfied, n_b_dissatisfied = a_dissatisfaction.sum(),b_dissatisfaction.sum()
    return (n_a_dissatisfied+n_b_dissatisfied)/np.size(M)

def mean_interratial_pears(M,boundary='wrap'):
    Kws = dict(mode='same',boundary=boundary)
    a_neights = convolve(M == 0,Kernel,**Kws)
    b_neights = convolve(M == 1,Kernel,**Kws)
    a_positions = np.argwhere(M == 1) 
    Y = np.transpose(a_positions)[0]
    X = np.transpose(a_positions)[1]
    b_neights_pears = b_neights[Y,X]
    b_positions = np.argwhere(M == 0)
    Y = np.transpose(b_positions)[0]
    X = np.transpose(b_positions)[1]
    a_neight_pears = a_neights
    interratial_pears = b_neights_pears.sum() + a_neight_pears.sum()
    return (interratial_pears/(np.size(M)*8))

def start(arg):
    M = rand_init(N,empty,A_to_B)
    similarity_1 = get_mean_similarity_ratio(M)
    dissatisfacton_1 = get_mean_dissatisfaction(M)
    mean_interratial_1 = mean_interratial_pears(M)
    bloked = False
    blocks_a = False
    blocks_b = False
    counter = 0
    dissatisfaction_n = 0
    for i in range(30000):
        M,dissatisfaction_n,*c = evolve(M,bloked,blocks_a,blocks_b)
        bloked,blocks_a,blocks_b=c
        counter = i+1
        if (dissatisfaction_n == 0 or bloked == True ) :
            break
    
    similarity = get_mean_similarity_ratio(M)
    dissatisfacton = get_mean_dissatisfaction(M)
    mean_interratial = mean_interratial_pears(M)
    return similarity_1,dissatisfacton_1,mean_interratial_1,similarity,dissatisfacton,mean_interratial,counter

if __name__ == '__main__':
    file_name = "schelling_values_100_model_2.csv"
    start_time = time.time()
    emptines = np.linspace(0.001,0.9,180)
    f = open(file_name, "w")
    f.write("vacant;similarity ratio inicial;mean dissatisfaction inicial;mean interratial pears inicial;similarity ratio final;mean dissatisfaction final;mean interratial pears final;number of iterations")
    f.close
    for emptys in emptines:
        with Pool(os.cpu_count(),initializer=inicialize_empty, initargs=(emptys,)) as p:
            sim1= p.imap(start,range(100))
            for i in zip(sim1):
                f = open(file_name, "a")
                f.write("\n")
                f.write("{};{};{};{};{};{};{};{}".format(emptys,i[0][0],i[0][1],i[0][2],i[0][3],i[0][4],i[0][5],i[0][6]))
                f.close
        f = open(file_name, "a")
        f.write("\n")
        f.write("\n")
        f.close   
        print(emptys)
                
        
    print("--- %s seconds ---" % (time.time() - start_time))