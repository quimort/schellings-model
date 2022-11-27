from itertools import count
import numpy as np
from scipy.signal import convolve2d as convolve
import time
from multiprocessing import Pool
import os
# Gloval variables of the simulation
N = 25
sim_t = 0.5
empty = 0.001
A_to_B = 1
Kernel = np.array([[1,1,1],[1,0,1],[1,1,1]],dtype=np.int8)
epsilon = 0.00001
Kernel2 = np.array([[2,-2],[2,-1],[2,0],[2,1],[2,2],[1,-2],[1,-1],[1,0],[1,1],[1,2],[0,-2],[0,-1],[0,0],[0,1],[0,2],[-1,-2],[-1,-1],[-1,0],[-1,1],[-1,2]\
    ,[-2,-2],[-2,-1],[-2,0],[-2,1],[-2,2]])

def rand_init(N,empty,a_to_b):
    """ Random grid initialitzation
    A = 0
    B = 1
    empty = -1
    """
    vacant = int(round(N*N*empty))
    population = N*N-vacant
    A = int(population*1/(1+1/a_to_b))
    B = int(population-A)
    M =np.zeros(int(N*N),dtype=np.int8)
    M[:B] = 1
    M[-vacant:] = -1
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
    old_a_neights = (a_neights[Y,X].reshape(vacant.shape[0],5,5))[:,1:4,1:4]
    old_b_neights = (b_neights[Y,X].reshape(vacant.shape[0],5,5))[:,1:4,1:4]
    old_neights = (neights[Y,X].reshape(vacant.shape[0],5,5))[:,1:4,1:4]
    if_type_a_dissatisfied = (position_a_neights < sim_t*position_all_neights)&(positon_type == 0)\
        &(old_a_neights >= sim_t*old_neights)
    if_type_a_dissatisfied = if_type_a_dissatisfied.reshape(vacant.shape[0],9)
    a_dissatysfied = np.apply_along_axis(calc_type_dissatisfyed,-1,if_type_a_dissatisfied)
    if_type_b_dissatisfied = (position_b_neights < sim_t*position_all_neights)&(positon_type == 1)\
        &(old_b_neights >= sim_t*old_neights)
    if_type_b_dissatisfied = if_type_b_dissatisfied.reshape(vacant.shape[0],9)
    b_dissatysfied = np.apply_along_axis(calc_type_dissatisfyed,-1,if_type_b_dissatisfied)
    dissatisfaied_vacant = a_dissatysfied + b_dissatysfied
    
    return dissatisfaied_vacant
    

def evolve(M,bloked,blocks,boundary='wrap'):
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
        return M,dissatisfaction_n,bloked,blocks
    random_number = np.random.randint(np.size(cordenates,axis=0),size=1)
    random_index = cordenates[random_number][0]
    index_vacants = np.argwhere(M == -1)
    agent_tipe = M[random_index[0]][random_index[1]]
    Y = np.transpose(index_vacants)[0]
    X = np.transpose(index_vacants)[1]
    if(np.size(index_vacants,axis=0)==0):
        print((M==-1).sum())
    if (agent_tipe == 0 ):
        dissatisfaied_vacant = check_happines_neighborhod(M,index_vacants,agent_tipe,random_index,a_neights,b_neights,neights)
        a_neights_vacants = a_neights[Y,X]
        neights_vacants = neights[Y,X]
        satisfaying_vacants_a = (a_neights_vacants >= sim_t*neights_vacants)
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
        satisfaying_vacants_b = (b_neights_vacants >= sim_t*neights_vacants)
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
            dissatisfaied_vacant = check_happines_neighborhod(M,index_vacants,agent_tipe,random_index,a_neights,b_neights,neights)
            satisfaying_vacants_b = (b_neights_vacants >= sim_t*neights_vacants)
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
            dissatisfaied_vacant = check_happines_neighborhod(M,index_vacants,agent_tipe,random_index,a_neights,b_neights,neights)
            satisfaying_vacants_a = (a_neights_vacants >= sim_t*neights_vacants)
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
    return M,dissatisfaction_n,bloked,blocks

def get_mean_similarity_ratio(M,empty,boundary='wrap'):

    Kws = dict(mode='same',boundary=boundary)
    a_neights = convolve(M == 0,Kernel,**Kws)
    b_neights = convolve(M == 1,Kernel,**Kws)
    neights = convolve(M != -1,Kernel,**Kws)
    a_neights = a_neights + epsilon
    b_neights = b_neights + epsilon
    neights = neights + epsilon
    n_similar_a = np.where(np.logical_and(np.logical_and(M !=-1,M == 0),neights != 0),\
        a_neights/8,0)
    n_similar_b = np.where(np.logical_and(np.logical_and(M !=-1,M == 1),neights != 0),\
         b_neights/8,0)
    n_similar = np.sum((n_similar_a+n_similar_b))
    return n_similar/((1-empty)*N*N)

def get_mean_dissatisfaction(M,empty,boundary='wrap'):

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
    return (n_a_dissatisfied+n_b_dissatisfied)/((1-empty)*N*N)

def mean_interratial_pears(M,boundary='wrap'):
    Kws = dict(mode='same',boundary=boundary)
    a_neights = convolve(M == 0,Kernel,**Kws)
    b_neights = convolve(M == 1,Kernel,**Kws)
    a_positions = np.argwhere(M == 1) 
    Y = np.transpose(a_positions)[0]
    X = np.transpose(a_positions)[1]
    b_neights_pears = b_neights[Y,X]
    a_neight_pears = a_neights[Y,X]
    interratial_pears = ((b_neights_pears.sum())/(b_neights_pears.sum()+a_neight_pears.sum()))
    return (interratial_pears)

def start(arg):
    M = rand_init(N,empty,A_to_B)
    similarity_1 = get_mean_similarity_ratio(M,empty)
    dissatisfacton_1 = get_mean_dissatisfaction(M,empty)
    mean_interratial_1 = mean_interratial_pears(M)
    bloked = False
    blocks = np.array([False,False])
    counter = 0
    
    for i in range(30000):
        M,dissatisfaction_n,bloked,blocks = evolve(M,bloked,blocks)
        counter = i+1
        if (dissatisfaction_n == 0 or bloked == True ) :
            break
    
    similarity = get_mean_similarity_ratio(M,empty)
    dissatisfacton = get_mean_dissatisfaction(M,empty)
    mean_interratial = mean_interratial_pears(M)
    return similarity_1,dissatisfacton_1,mean_interratial_1,similarity,dissatisfacton,mean_interratial,counter
def inicialize_empty(emptines):
    global empty

    empty = emptines
if __name__ == '__main__':
    file_name = "schelling_values_1000_model_3_25.csv"
    start_time = time.time()
    emptines = np.logspace(-3,0,180)
    f = open(file_name, "w")
    f.write("vacant;similarity ratio inicial;mean dissatisfaction inicial;mean interratial pears inicial;similarity ratio final;mean dissatisfaction final;mean interratial pears final;number of iterations")
    for emptys in emptines:
        with Pool(os.cpu_count(),initializer=inicialize_empty, initargs=(emptys,)) as p:
            sim1= p.imap(start,range(1000))
            for i in zip(sim1):
                f.write("\n")
                f.write("{};{};{};{};{};{};{};{}".format(emptys,i[0][0],i[0][1],i[0][2],i[0][3],i[0][4],i[0][5],i[0][6]))
        f = open(file_name, "a")
        f.write("\n")
        f.write("\n")  
        print(emptys)
    f.close()      
              
        
    print("--- %s seconds ---" % (time.time() - start_time))