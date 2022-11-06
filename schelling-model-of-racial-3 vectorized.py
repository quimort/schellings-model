from itertools import count
import numpy as np
from scipy.signal import convolve2d as convolve
import time
# Gloval variables of the simulation
N = 50
sim_t = 0.5
empty = 0.7
A_to_B = 1
Kernel = np.array([[1,1,1],[1,0,1],[1,1,1]],dtype=np.int8)
epsilon = 0.00001
bloked = False
blocks= np.array([False,False])
Kernel2 = np.array([[2,-2],[2,-1],[2,0],[2,1],[2,2],[1,-2],[1,-1],[1,0],[1,1],[1,2],[0,-2],[0,-1],[0,0],[0,1],[0,2],[-1,-2],[-1,-1],[-1,0],[-1,1],[-1,2]\
    ,[-2,-2],[-2,-1],[-2,0],[-2,1],[-2,2]])

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

def calculete_vacant_neightbours(vacant):
    return Kernel2 + vacant

def change_type(positions,type):
    positions2 = positions.reshape(5,5)
    positions2[2][2] = type
    return positions2.reshape(positions.shape[0])

def calc_neights(positions,type,boundary='wrap'):
    Kws = dict(mode='same',boundary=boundary)
    positions = positions.reshape(5,5)
    positions2 = (convolve(positions == type,Kernel,**Kws))[1:4,1:4]
    return positions2

def calc_all_neights(positions,boundary='wrap'):
    Kws = dict(mode='same',boundary=boundary)
    positions = positions.reshape(5,5)
    positions2 = (convolve(positions != -1,Kernel,**Kws))[1:4,1:4]
    return positions2
def change_pos_type(positions):
    positions = positions.reshape(5,5)[1:4,1:4]
    return positions
def calc_type_dissatisfyed(positions):
    value = False
    if(True in positions):
        value = True
    return value


def check_happines_neighborhod(M,new_positions,type,old_position,a_neights,b_neights,neights,boundary='wrap'):
    Kws = dict(mode='same',boundary=boundary)
    vacant = np.apply_along_axis(calculete_vacant_neightbours,-1,new_positions)
    vacant = np.where(vacant == np.size(M,axis=0),0,vacant)
    vacant = np.where(vacant > np.size(M,axis=0),1,vacant)
    vacant2 = vacant.reshape(-1, vacant.shape[-1])
    Y = np.transpose(vacant2)[0]
    X = np.transpose(vacant2)[1]
    positon_type = M[Y,X]
    positon_type = positon_type.reshape(vacant.shape[0],vacant.shape[1])
    positon_type = np.apply_along_axis(change_type,-1,positon_type,type)
    position_a_neights = np.apply_along_axis(calc_neights,-1,positon_type,0)
    position_b_neights = np.apply_along_axis(calc_neights,-1,positon_type,1)
    position_all_neights = np.apply_along_axis(calc_all_neights,-1,positon_type)
    positon_type = np.apply_along_axis(change_pos_type,-1,positon_type)
    position_a_neights = position_a_neights +epsilon
    position_b_neights = position_b_neights +epsilon
    position_all_neights = position_all_neights +epsilon
    old_a_neights = (a_neights[Y,X].reshape(vacant.shape[0],5,5))[:,1:4,1:4]
    old_b_neights = (b_neights[Y,X].reshape(vacant.shape[0],5,5))[:,1:4,1:4]
    old_neights = (neights[Y,X].reshape(vacant.shape[0],5,5))[:,1:4,1:4]
    if_type_a_dissatisfied = (position_a_neights/position_all_neights < sim_t)&(positon_type == 0)\
        &(old_a_neights/old_neights >= sim_t)
    if_type_a_dissatisfied = if_type_a_dissatisfied.reshape(vacant.shape[0],9)
    a_dissatysfied = np.apply_along_axis(calc_type_dissatisfyed,-1,if_type_a_dissatisfied)
    if_type_b_dissatisfied = (position_b_neights/position_all_neights < sim_t)&(positon_type == 1)\
        &(old_b_neights/old_neights >= sim_t)
    if_type_b_dissatisfied = if_type_b_dissatisfied.reshape(vacant.shape[0],9)
    b_dissatysfied = np.apply_along_axis(calc_type_dissatisfyed,-1,if_type_b_dissatisfied)
    dissatisfaied_vacant = a_dissatysfied + b_dissatysfied
    
    return dissatisfaied_vacant
    

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
        dissatisfaied_vacant = check_happines_neighborhod(M,index_vacants,agent_tipe,random_index,a_neights,b_neights,neights)
        a_neights_vacants = a_neights[Y,X]
        neights_vacants = neights[Y,X]
        a_neights_vacants = a_neights_vacants + epsilon
        neights_vacants = neights_vacants +epsilon
        satisfaying_vacants_a = (a_neights_vacants/neights_vacants >= sim_t)
        satisfaying_vacants_a = (satisfaying_vacants_a == True)&(dissatisfaied_vacant == False)
        if(True in satisfaying_vacants_a):
            array_of_good_vacants = np.where(satisfaying_vacants_a == True)
            move_to = index_vacants[array_of_good_vacants[0][0]]
            M[random_index[0]][random_index[1]] = -1
            M[move_to[0]][move_to[1]] = 0
        if( True not in satisfaying_vacants_a):
            blocks[0] = True
    else:
        dissatisfaied_vacant = check_happines_neighborhod(M,index_vacants,agent_tipe,random_index,a_neights,b_neights,neights)
        b_neights_vacants = b_neights[Y,X]
        neights_vacants = neights[Y,X]
        b_neights_vacants = b_neights_vacants + epsilon
        neights_vacants = neights_vacants +epsilon
        satisfaying_vacants_b = (b_neights_vacants/neights_vacants >= sim_t)
        satisfaying_vacants_b = (satisfaying_vacants_b == True)&(dissatisfaied_vacant == False)
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
            dissatisfaied_vacant = check_happines_neighborhod(M,index_vacants,agent_tipe,random_index,a_neights,b_neights,neights)
            satisfaying_vacants_b = (b_neights_vacants/neights_vacants >= sim_t)
            satisfaying_vacants_b = (satisfaying_vacants_b == True)&(dissatisfaied_vacant == False)
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
            dissatisfaied_vacant = check_happines_neighborhod(M,index_vacants,agent_tipe,random_index,a_neights,b_neights,neights)
            satisfaying_vacants_a = (a_neights_vacants/neights_vacants >= sim_t)
            satisfaying_vacants_a = (satisfaying_vacants_a == True)&(dissatisfaied_vacant == False)
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
emptines = np.linspace(0.021,0.9,180)
for i in emptines:
    empty = i
    for ii in range(100):
        M = rand_init(N,empty,A_to_B)
        similarity_1 = get_mean_similarity_ratio(M)
        dissatisfacton_1 = get_mean_dissatisfaction(M)
        mean_interratial_1 = mean_interratial_pears(M)
        continua = True
        bloked = False
        blocks[0] = False
        blocks[1] = False
        counter = 0
        while(continua):
            M,dissatisfaction_n = evolve(M)
            counter += 1
            if (dissatisfaction_n == 0 or bloked == True):
                continua = False
        similarity = get_mean_similarity_ratio(M)
        dissatisfacton = get_mean_dissatisfaction(M)
        mean_interratial = mean_interratial_pears(M)
        f = open("schelling_values_100.csv", "a")
        f.write("\n")
        f.write("{};{};{};{};{};{};{};{}".format(empty,similarity_1,dissatisfacton_1,mean_interratial_1,similarity,dissatisfacton,mean_interratial,counter))
        print(ii)
    f = open("schelling_values_100.csv", "a")
    f.write("\n")
    f.write("\n")
    f.close
    print(empty)
