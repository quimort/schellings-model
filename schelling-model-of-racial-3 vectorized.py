from itertools import count
import numpy as np
from scipy.signal import convolve2d as convolve
import time
# Gloval variables of the simulation
N = 60
sim_t = 0.5
empty = 0.01
A_to_B = 1
Kernel = np.array([[1,1,1],[1,0,1],[1,1,1]],dtype=np.int8)
epsilon = 0.00001
bloked = False
blocks= np.array([False,False])

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

def check_happines_neighborhod(M,new_positions,type,old_position,boundary='wrap'):
    Kws = dict(mode='same',boundary=boundary)
    valid_vacant = np.zeros(np.size(new_positions),dtype=bool)
    contador = 0
    for vacant in new_positions:
        new_M = M
        new_M[vacant[0]][vacant[1]] = type
        new_M[old_position[0]][old_position[1]] = -1
        a_neights = convolve(new_M == 0,Kernel,**Kws)
        b_neights = convolve(new_M == 1,Kernel,**Kws)
        neights = convolve(new_M != -1,Kernel,**Kws)
        Kernel2 = np.array([[[1,-1],[1,0],[1,1]],[[0,-1],[0,0],[0,1]],[[-1,-1],[-1,0],[-1,1]]])
        possible_neights = Kernel2 + vacant
        Y = np.transpose(possible_neights)[0]
        X = np.transpose(possible_neights)[1]
        positon_type = new_M[Y,X]
        position_a_neights = a_neights[Y,X]
        position_b_neights = b_neights[Y,X]
        position_all_neights = neights[Y,X]
        if_type_a_satisfied = (position_a_neights/position_all_neights >= sim_t)&(positon_type == 0)
        if_type_b_satisfied = (position_b_neights/position_all_neights >= sim_t)&(positon_type == 0)
        satisfactory = (if_type_a_satisfied == True)|(if_type_b_satisfied == True)
        if(False not in satisfactory):
            valid_vacant[contador] = True 
        contador += 1
    
    return valid_vacant
    

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
    global blocks 
    global bloked 
    a_neights = convolve(M == 0,Kernel,**Kws)
    b_neights = convolve(M == 1,Kernel,**Kws)
    neights = convolve(M != -1,Kernel,**Kws)
    a_neights = a_neights + epsilon
    b_neights = b_neights + epsilon
    neights = neights + epsilon
    a_dissatisfaction = (a_neights/neights < sim_t)&(M == 0)
    b_dissatisfaction = (b_neights/neights < sim_t)&(M == 1)
    n_a_dissatisfied, n_b_dissatisfied = a_dissatisfaction.sum(),b_dissatisfaction.sum()
    dissatisfaction_n = (n_a_dissatisfied+n_b_dissatisfied)/np.size(M)
    cordenates_a = np.argwhere(a_dissatisfaction)
    cordenates_b = np.argwhere(b_dissatisfaction)
    cordenates = np.concatenate((cordenates_a,cordenates_b),axis = 0)
    if (np.size(cordenates,axis=0) == 0):
        bloked = True
        return M,dissatisfaction_n
    random_number = np.random.randint(np.size(cordenates,axis=0),size=1)
    random_index = cordenates[random_number][0]
    index_vacants = np.argwhere(M == -1)
    agent_tipe = M[random_index[0]][random_index[1]]
    Y = np.transpose(index_vacants)[0]
    X = np.transpose(index_vacants)[1]
    if (agent_tipe == 0 ):
        valid_vacant = check_happines_neighborhod(M,index_vacants,agent_tipe,random_index)
        a_neights_vacants = a_neights[Y,X]
        neights_vacants = neights[Y,X]
        a_neights_vacants = a_neights_vacants + epsilon
        neights_vacants = neights_vacants +epsilon
        satisfaying_vacants_a = (a_neights_vacants/neights_vacants >= sim_t)
        satisfaying_vacants_a = (satisfaying_vacants_a)&(valid_vacant)
        if(True in satisfaying_vacants_a):
            array_of_good_vacants = np.where(satisfaying_vacants_a == True)
            move_to = index_vacants[array_of_good_vacants[0][0]]
            M[random_index[0]][random_index[1]] = -1
            M[move_to[0]][move_to[1]] = 0
        if( True not in satisfaying_vacants_a):
            blocks[0] = True
    else:
        valid_vacant = check_happines_neighborhod(M,index_vacants,agent_tipe,random_index)
        b_neights_vacants = b_neights[Y,X]
        neights_vacants = neights[Y,X]
        b_neights_vacants = b_neights_vacants + epsilon
        neights_vacants = neights_vacants +epsilon
        satisfaying_vacants_b = (b_neights_vacants/neights_vacants >= sim_t)
        satisfaying_vacants_b = (satisfaying_vacants_b)&(valid_vacant)
        if(True in satisfaying_vacants_b):
            array_of_good_vacants = np.where(satisfaying_vacants_b == True)
            move_to = index_vacants[array_of_good_vacants[0][0]]
            M[random_index[0]][random_index[1]] = -1
            M[move_to[0]][move_to[1]] = 1
        if( True not in satisfaying_vacants_b):
            blocks[1] = True
    if(blocks[0] == True):
        "a agents are blocked"
        index_test = cordenates_b
        if(np.size(index_test,axis=0) != 0):
            cordenate_test = index_test[0]
            b_neights_vacants = b_neights[Y,X]
            neights_vacants = neights[Y,X]
            b_neights_vacants = b_neights_vacants + epsilon
            neights_vacants = neights_vacants +epsilon
            valid_vacant = check_happines_neighborhod(M,index_vacants,agent_tipe,random_index)
            satisfaying_vacants_b = (b_neights_vacants/neights_vacants >= sim_t)
            satisfaying_vacants_b = (satisfaying_vacants_b)&(valid_vacant)
            if(True in satisfaying_vacants_b):
                array_of_good_vacants = np.where(satisfaying_vacants_b == True)
                move_to = index_vacants[array_of_good_vacants[0][0]]
                M[cordenate_test[0]][cordenate_test[1]] = -1
                M[move_to[0]][move_to[1]] = 1
            if( True not in satisfaying_vacants_b):
                blocks[1] = True
    if(blocks[1] == True):
        "b agents are blocked"
        index_test = cordenates_a
        if(np.size(index_test,axis=0) != 0):
            cordenate_test = index_test[0]
            a_neights_vacants = a_neights[Y,X]
            neights_vacants = neights[Y,X]
            a_neights_vacants = a_neights_vacants + epsilon
            neights_vacants = neights_vacants +epsilon
            valid_vacant = check_happines_neighborhod(M,index_vacants,agent_tipe,random_index)
            satisfaying_vacants_a = (a_neights_vacants/neights_vacants >= sim_t)
            satisfaying_vacants_a = (satisfaying_vacants_a)&(valid_vacant)
            if(True in satisfaying_vacants_a):
                array_of_good_vacants = np.where(satisfaying_vacants_a == True)
                move_to = index_vacants[array_of_good_vacants[0][0]]
                M[cordenate_test[0]][cordenate_test[1]] = -1
                M[move_to[0]][move_to[1]] = 1
            if( True not in satisfaying_vacants_a):
                blocks[0] = True
    if(blocks[0] == True and blocks[1] == True):
       bloked= True
    return M,dissatisfaction_n

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
start_time = time.time()
M = rand_init(N,empty,A_to_B)
similarity = get_mean_similarity_ratio(M)
dissatisfacton = get_mean_dissatisfaction(M)
print("similarity initial = {} /dissatisfaction initial = {}".format(similarity,dissatisfacton))
continua = True
counter = 0
while(continua):
    M,dissatisfaction_n = evolve(M)
    counter += 1
    if (dissatisfaction_n == 0 or bloked == True):
        continua = False
similarity = get_mean_similarity_ratio(M)
dissatisfacton = get_mean_dissatisfaction(M)
mean_interratial = mean_interratial_pears(M)
print("similarity final = {} /dissatisfaction final = {} / mean inerratial pears = {}"\
    .format(similarity,dissatisfacton,mean_interratial))
print("number of iterations = {}".format(counter))
print("--- %s seconds ---" % (time.time() - start_time))

