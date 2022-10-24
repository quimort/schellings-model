import numpy as np
from scipy.signal import convolve2d as convolve

# Gloval variables of the simulation
N = 60
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

def check_happines_neighborhod(M,new_position,type,old_position,boundary='wrap'):
    Kws = dict(mode='same',boundary=boundary)
    new_M = M
    new_M[new_position[0]][new_position[1]] = type
    new_M[old_position[0]][old_position[1]] = -1
    a_neights = convolve(M == 0,Kernel,**Kws)
    b_neights = convolve(M == 1,Kernel,**Kws)
    neights = convolve(M != -1,Kernel,**Kws)
    new_a_neights = convolve(new_M == 0,Kernel,**Kws)
    new_b_neights = convolve(new_M == 1,Kernel,**Kws)
    new_neights = convolve(new_M != -1,Kernel,**Kws)
    Kernel2 = np.array([[1,-1],[1,0],[1,1],[0,-1],[0,1],[-1,-1],[-1,0],[-1,1]])
    possible_neights = Kernel2 + new_position
    possible_neights = np.where(possible_neights >= np.size(M,axis=0),0,possible_neights)
    dissatisfaied = False
    counter = 0
    while (dissatisfaied != True) and (counter < np.size(possible_neights,axis=0)):
        neight_type = new_M[possible_neights[counter][0]][possible_neights[counter][1]]
        if (neight_type == 0):
            new_position_a_neights = new_a_neights[possible_neights[counter][0]][possible_neights[counter][1]]
            new_position_neights = new_neights[possible_neights[counter][0]][possible_neights[counter][1]]
            position_a_neights = a_neights[possible_neights[counter][0]][possible_neights[counter][1]]
            position_neights = neights[possible_neights[counter][0]][possible_neights[counter][1]]
            dissatisfaied = (new_position_a_neights/new_position_neights < sim_t)&(position_a_neights/position_neights > sim_t)
        elif (neight_type == 1):
            new_position_b_neights = new_b_neights[possible_neights[counter][0]][possible_neights[counter][1]]
            new_position_neights = new_neights[possible_neights[counter][0]][possible_neights[counter][1]]
            position_b_neights = b_neights[possible_neights[counter][0]][possible_neights[counter][1]]
            position_neights = neights[possible_neights[counter][0]][possible_neights[counter][1]]
            dissatisfaied = (new_position_b_neights/new_position_neights <= sim_t)&(position_b_neights/position_neights > sim_t)
        counter += 1
    
    return dissatisfaied
    




    

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
        neight_dissatisfaied = check_happines_neighborhod(M,index_vacants[counter],agent_tipe,random_index)
        if (agent_tipe == 0):
            position_a_neights = a_neights[index_vacants[counter][0]][index_vacants[counter][1]]
            position_neights = neights[index_vacants[counter][0]][index_vacants[counter][1]]
            if (position_a_neights/position_neights >= sim_t) and (neight_dissatisfaied == False):
                M[random_index[0]][random_index[1]] = -1
                M[index_vacants[counter][0]][index_vacants[counter][1]] = 0
                foundit = True
        else:
            position_b_neights = b_neights[index_vacants[counter][0]][index_vacants[counter][1]]
            position_neights = neights[index_vacants[counter][0]][index_vacants[counter][1]]
            if (position_b_neights/position_neights >= sim_t) and (neight_dissatisfaied == False):
                M[random_index[0]][random_index[1]] = -1
                M[index_vacants[counter][0]][index_vacants[counter][1]] = 0
                foundit = True
        counter += 1

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


f = open("valuesfor_01.csv", "w")
f.write("similarity ratio;mean dissatisfaction")
f.close
for ii in range(0,100):
    M = rand_init(N,empty,A_to_B)
    similarity = get_mean_similarity_ratio(M)
    dissatisfacton = get_mean_dissatisfaction(M)
    for i in range(5100):
        M = evolve(M)
        if (get_mean_dissatisfaction(M) == 0):
            break
    similarity = get_mean_similarity_ratio(M)
    dissatisfacton = get_mean_dissatisfaction(M)
    f = open("valuesfor_01.csv", "a")
    f.write("\n")
    f.write("{};{}".format(similarity,dissatisfacton))
    f.close


