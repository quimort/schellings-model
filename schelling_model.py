import numpy as np
from scipy.signal import convolve2d as convolve

class _ComonOperations:
    def __init__(self,N,sim_t,empty,ratio,kernel,epsilon,second_kernel):

        self.n = N
        self.sim_t = sim_t
        self.empty = empty
        self.ratio = ratio
        self.kernel = kernel
        self.epsilon = epsilon
        self.second_kernel = second_kernel
        self.blocked = False
        self.blocks = np.array([False,False])

    def _calculete_vacant_neightbours(self,vacant):
        return self.second_kernel + vacant
    
    def _change_type(self,positions,type):
        positions2 = positions.reshape(5,5)
        positions2[2][2] = type
        return positions2.reshape(positions.shape[0])
    
    def _calc_neights(self,positions,type,boundary='wrap'):
        Kws = dict(mode='same',boundary=boundary)
        positions = positions.reshape(5,5)
        positions2 = (convolve(positions == type,self.kernel,**Kws))[1:4,1:4]
        return positions2

    def _calc_all_neights(self,positions,boundary='wrap'):
        Kws = dict(mode='same',boundary=boundary)
        positions = positions.reshape(5,5)
        positions2 = (convolve(positions != -1,self.kernel,**Kws))[1:4,1:4]
        return positions2
    
    def _change_pos_type(self,positions):
        positions = positions.reshape(5,5)[1:4,1:4]
        return positions
    
    def _calc_type_dissatisfyed(self,positions):
        value = False
        if(True in positions):
            value = True
        return value
    
    def check_happines_neighborhod(self,M,new_positions,type,old_position,a_neights,b_neights,neights,boundary='wrap'):
        vacant = np.apply_along_axis(self._calculete_vacant_neightbours,-1,new_positions)
        vacant = np.where(vacant == np.size(M,axis=0),0,vacant)
        vacant = np.where(vacant > np.size(M,axis=0),1,vacant)
        vacant2 = vacant.reshape(-1, vacant.shape[-1])
        Y = np.transpose(vacant2)[0]
        X = np.transpose(vacant2)[1]
        positon_type = M[Y,X]
        positon_type = positon_type.reshape(vacant.shape[0],vacant.shape[1])
        positon_type = np.apply_along_axis(self._change_type,-1,positon_type,type)
        position_a_neights = np.apply_along_axis(self._calc_neights,-1,positon_type,0)
        position_b_neights = np.apply_along_axis(self._calc_neights,-1,positon_type,1)
        position_all_neights = np.apply_along_axis(self._calc_all_neights,-1,positon_type)
        positon_type = np.apply_along_axis(self._change_pos_type,-1,positon_type)
        old_a_neights = (a_neights[Y,X].reshape(vacant.shape[0],5,5))[:,1:4,1:4]
        old_b_neights = (b_neights[Y,X].reshape(vacant.shape[0],5,5))[:,1:4,1:4]
        old_neights = (neights[Y,X].reshape(vacant.shape[0],5,5))[:,1:4,1:4]
        if_type_a_dissatisfied = (position_a_neights < self.sim_t*position_all_neights)&(positon_type == 0)\
            &(old_a_neights >= self.sim_t*old_neights)
        if_type_a_dissatisfied = if_type_a_dissatisfied.reshape(vacant.shape[0],9)
        a_dissatysfied = np.apply_along_axis(self._calc_type_dissatisfyed,-1,if_type_a_dissatisfied)
        if_type_b_dissatisfied = (position_b_neights < self.sim_t*position_all_neights)&(positon_type == 1)\
            &(old_b_neights >= self.sim_t*old_neights)
        if_type_b_dissatisfied = if_type_b_dissatisfied.reshape(vacant.shape[0],9)
        b_dissatysfied = np.apply_along_axis(self._calc_type_dissatisfyed,-1,if_type_b_dissatisfied)
        dissatisfaied_vacant = a_dissatysfied + b_dissatysfied
        
        return dissatisfaied_vacant
    
    def get_mean_similarity_ratio(self,M,empty,boundary='wrap'):

        Kws = dict(mode='same',boundary=boundary)
        a_neights = convolve(M == 0,self.kernel,**Kws)
        b_neights = convolve(M == 1,self.kernel,**Kws)
        neights = convolve(M != -1,self.kernel,**Kws)
        a_neights = a_neights + self.epsilon
        b_neights = b_neights + self.epsilon
        neight_ = np.copy(neights)
        neights = neights + self.epsilon
        n_similar_a = (a_neights/neights)*(neight_!=0)*(M==0)
        n_similar_b = (b_neights/neights)*(neight_!=0)*(M==1)
        n_similar = int(np.sum((n_similar_a+n_similar_b)))
        no_neights = (neight_ == 0)*(M!=-1)
        no_neights_val = np.sum(no_neights)
        return n_similar/((1-empty)*self.n*self.n-no_neights_val)

    def get_mean_dissatisfaction(self,M,empty,boundary='wrap'):

        Kws = dict(mode='same',boundary=boundary)
        a_neights = convolve(M == 0,self.kernel,**Kws)
        b_neights = convolve(M == 1,self.kernel,**Kws)
        neights = convolve(M != -1,self.kernel,**Kws)
        a_neights = a_neights + self.epsilon
        b_neights = b_neights + self.epsilon
        neights = neights + self.epsilon
        a_dissatisfaction = (a_neights/neights < self.sim_t)&(M == 0)
        b_dissatisfaction = (b_neights/neights < self.sim_t)&(M == 1)
        n_a_dissatisfied, n_b_dissatisfied = a_dissatisfaction.sum(),b_dissatisfaction.sum()
        return (n_a_dissatisfied+n_b_dissatisfied)/((1-empty)*self.n*self.n)

    def mean_interratial_pears(self,M,boundary='wrap'):
        Kws = dict(mode='same',boundary=boundary)
        a_neights = convolve(M == 0,self.kernel,**Kws)
        b_neights = convolve(M == 1,self.kernel,**Kws)
        a_positions = np.argwhere(M == 0) 
        Y = np.transpose(a_positions)[0]
        X = np.transpose(a_positions)[1]
        b_neights_pears = b_neights[Y,X]
        a_neight_pears = a_neights[Y,X]
        interratial_pears_a = ((b_neights_pears.sum())/(b_neights_pears.sum()+a_neight_pears.sum()))
        b_positions = np.argwhere(M == 1) 
        Y = np.transpose(b_positions)[0]
        X = np.transpose(b_positions)[1]
        a_neights_pears = a_neights[Y,X]
        b_neights_pears = b_neights[Y,X]
        interratial_pears_b = ((a_neights_pears.sum())/(b_neights_pears.sum()+a_neights_pears.sum()))
        return (interratial_pears_a,interratial_pears_b,(interratial_pears_a+interratial_pears_b))

class AltruisticSchellingModel:

    def __init__(self,N,sim_t,empty,ratio,kernel,epsilon,second_kernel,boundary='wrap'):
        
        self.n = N
        self.sim_t = sim_t
        self.empty = empty
        self.ratio = ratio
        self.kernel = kernel
        self.epsilon = epsilon
        self.second_kernel = second_kernel
        self.boundary = boundary
        self.blocked = False
        self.blocks = np.array([False,False])
        self.M = None
        self._ComonOperatios = _ComonOperations(self.n,self.sim_t,self.empty,self.ratio,self.kernel,self.epsilon,self.second_kernel)
    
    def rand_init(self):
        """ Random grid initialitzation
        A = 0
        B = 1
        empty = -1
        """
        vacant = int(round(self.n*self.n*self.empty))
        population = self.n*self.n-vacant
        A = int(population*1/(1+1/self.ratio))
        B = int(population-A)
        M =np.zeros(int(self.n*self.n),dtype=np.int8)
        M[:B] = 1
        M[-vacant:] = -1
        np.random.shuffle(M)
        self.M =  M.reshape(int(self.n),int(self.n))

        return self
    
    def step(self):
        """
        Args:
            M(numpy.array): the matrix to be evolved
            boundary(str): Either wrap, fill or symm
        if the siilarity ratio of neighbours
        to the enitre neghtbourhood polupation
        is lower than sim_t,
        then the individual moves to an empty house. 
        """
        Kws = dict(mode='same',boundary=self.boundary)
        a_neights = convolve(self.M == 0,self.kernel,**Kws)
        b_neights = convolve(self.M == 1,self.kernel,**Kws)
        neights = convolve(self.M != -1,self.kernel,**Kws)
        a_dissatisfaction = (a_neights < self.sim_t*neights)&(self.M == 0)
        b_dissatisfaction = (b_neights < self.sim_t*neights)&(self.M == 1)
        cordenates_a = np.argwhere(a_dissatisfaction)
        cordenates_b = np.argwhere(b_dissatisfaction)
        cordenates = np.concatenate((cordenates_a,cordenates_b),axis = 0)
        if (np.size(cordenates,axis=0) == 0):

            return self
        
        random_number = np.random.randint(np.size(cordenates,axis=0),size=1)
        random_index = cordenates[random_number][0]
        index_vacants = np.argwhere(self.M == -1)
        agent_tipe = self.M[random_index[0]][random_index[1]]
        Y = np.transpose(index_vacants)[0]
        X = np.transpose(index_vacants)[1]
        if(np.size(index_vacants,axis=0)==0):
            print((self.M==-1).sum())
        if (agent_tipe == 0 ):
            dissatisfaied_vacant = self._ComonOperatios.check_happines_neighborhod(self.M,index_vacants,agent_tipe,random_index,a_neights,b_neights,neights)
            a_neights_vacants = a_neights[Y,X]
            neights_vacants = neights[Y,X]
            satisfaying_vacants_a = (a_neights_vacants >= self.sim_t*neights_vacants)
            satisfaying_vacants_a = (satisfaying_vacants_a == True)&(dissatisfaied_vacant == False)
            if(True in satisfaying_vacants_a):
                array_of_good_vacants = np.where(satisfaying_vacants_a == True)
                move_to = index_vacants[array_of_good_vacants[0][0]]
                self.M[random_index[0]][random_index[1]] = -1
                self.M[move_to[0]][move_to[1]] = 0
            if( True not in satisfaying_vacants_a):
                self.blocks[0] = True
        else:
            dissatisfaied_vacant = self._ComonOperatios.check_happines_neighborhod(self.M,index_vacants,agent_tipe,random_index,a_neights,b_neights,neights)
            b_neights_vacants = b_neights[Y,X]
            neights_vacants = neights[Y,X]
            satisfaying_vacants_b = (b_neights_vacants >= self.sim_t*neights_vacants)
            satisfaying_vacants_b = (satisfaying_vacants_b == True)&(dissatisfaied_vacant == False)
            if(True in satisfaying_vacants_b):
                array_of_good_vacants = np.where(satisfaying_vacants_b == True)
                move_to = index_vacants[array_of_good_vacants[0][0]]
                self.M[random_index[0]][random_index[1]] = -1
                self.M[move_to[0]][move_to[1]] = 1
            if( True not in satisfaying_vacants_b):
                self.blocks[1] = True
        if(self.blocks[0] == True):
            "a agents are self.blocked"
            index_test = cordenates_b
            if(np.size(index_test,axis=0) != 0):
                cordenate_test = index_test[0]
                b_neights_vacants = b_neights[Y,X]
                neights_vacants = neights[Y,X]
                dissatisfaied_vacant = self._ComonOperatios._check_happines_neighborhod(self.M,index_vacants,agent_tipe,random_index,a_neights,b_neights,neights)
                satisfaying_vacants_b = (b_neights_vacants >= self.sim_t*neights_vacants)
                satisfaying_vacants_b = (satisfaying_vacants_b == True)&(dissatisfaied_vacant == False)
                if(True in satisfaying_vacants_b):
                    array_of_good_vacants = np.where(satisfaying_vacants_b == True)
                    move_to = index_vacants[array_of_good_vacants[0][0]]
                    self.M[cordenate_test[0]][cordenate_test[1]] = -1
                    self.M[move_to[0]][move_to[1]] = 1
                if( True not in satisfaying_vacants_b):
                    self.blocks[1] = True
        if(self.blocks[1] == True):
            "b agents are blocked"
            index_test = cordenates_a
            if(np.size(index_test,axis=0) != 0):
                cordenate_test = index_test[0]
                a_neights_vacants = a_neights[Y,X]
                neights_vacants = neights[Y,X]
                dissatisfaied_vacant = self._ComonOperatios._check_happines_neighborhod(self.M,index_vacants,agent_tipe,random_index,a_neights,b_neights,neights)
                satisfaying_vacants_a = (a_neights_vacants >= self.sim_t*neights_vacants)
                satisfaying_vacants_a = (satisfaying_vacants_a == True)&(dissatisfaied_vacant == False)
                if(True in satisfaying_vacants_a):
                    array_of_good_vacants = np.where(satisfaying_vacants_a == True)
                    move_to = index_vacants[array_of_good_vacants[0][0]]
                    self.M[cordenate_test[0]][cordenate_test[1]] = -1
                    self.M[move_to[0]][move_to[1]] = 1
                if( True not in satisfaying_vacants_a):
                    self.blocks[0] = True
        if(self.blocks[0] == True and self.blocks[1] == True):
            self.bloked= True
        return self

    def get_matrix(self):

        return self.M
    
    def get_mean_similarity_ratio(self):
        return self._ComonOperatios.get_mean_similarity_ratio(self.M,self.empty,self.boundary)
    
    def get_mean_dissatisfaction(self):
        return self._ComonOperatios.get_mean_dissatisfaction(self.M,self.empty,self.boundary)
    
    def get_mean_interratial_pears(self):
        return self._ComonOperatios.mean_interratial_pears(self.M,self.boundary)
    
    def print_matrix(self):
        print(self.M)

class ClassicSchellingModel:

    def __init__(self,N,sim_t,empty,ratio,kernel,epsilon,boundary='wrap'):
        
        self.n = N
        self.sim_t = sim_t
        self.empty = empty
        self.ratio = ratio
        self.kernel = kernel
        self.epsilon = epsilon
        self.blocked = False
        self.blocks_a = False
        self.blocks_b = False
        self.boundary = boundary
        self._ComonOperatios = _ComonOperations(self.n,self.sim_t,self.empty,self.ratio,self.kernel,self.epsilon,self.kernel)

    def rand_init(self):
        """ Random grid initialitzation
        A = 0
        B = 1
        empty = -1
        """
        vacant = int(round(self.n*self.n*self.empty))
        population = self.n*self.n-vacant
        A = int(population*1/(1+1/self.ratio))
        B = int(population-A)
        M =np.zeros(int(self.n*self.n),dtype=np.int8)
        M[:B] = 1
        M[-vacant:] = -1
        np.random.shuffle(M)
        self.M = M.reshape(int(self.n),int(self.n))
        return  self
    
    def step(self):
        """
        Args:
            M(numpy.array): the matrix to be evolved
            boundary(str): Either wrap, fill or symm
        if the siilarity ratio of neighbours
        to the enitre neghtbourhood polupation
        is lower than sim_t,
        then the individual moves to an empty house. 
        """
        
        Kws = dict(mode='same',boundary=self.boundary)
        a_neights = convolve(self.M == 0,self.kernel,**Kws)
        b_neights = convolve(self.M == 1,self.kernel,**Kws)
        neights = convolve(self.M != -1,self.kernel,**Kws)
        a_dissatisfaction = (a_neights < self.sim_t*neights)&(self.M == 0)
        b_dissatisfaction = (b_neights < self.sim_t*neights)&(self.M == 1)
        cordenates_a = np.argwhere(a_dissatisfaction)
        cordenates_b = np.argwhere(b_dissatisfaction)
        cordenates = np.concatenate((cordenates_a,cordenates_b),axis = 0)
        if (np.size(cordenates,axis=0) == 0):
            
            return self
        
        random_number = np.random.randint(np.size(cordenates,axis=0),size=1)
        random_index = cordenates[random_number][0]
        index_vacants = np.argwhere(self.M == -1)
        agent_tipe = self.M[random_index[0]][random_index[1]]
        Y = np.transpose(index_vacants)[0]
        X = np.transpose(index_vacants)[1]
        if (agent_tipe == 0 ):
            a_neights_vacants = a_neights[Y,X]
            neights_vacants = neights[Y,X]
            satisfaying_vacants_a = (a_neights_vacants >= self.sim_t*neights_vacants)
            if(True in satisfaying_vacants_a):
                array_of_good_vacants = np.where(satisfaying_vacants_a == True)
                move_to = index_vacants[array_of_good_vacants[0][0]]
                self.M[random_index[0]][random_index[1]] = -1
                self.M[move_to[0]][move_to[1]] = 0
            if( True not in satisfaying_vacants_a):
                self.blocks_a = True
        else:
            b_neights_vacants = b_neights[Y,X]
            neights_vacants = neights[Y,X]
            satisfaying_vacants_b = (b_neights_vacants >= self.sim_t*neights_vacants)
            if(True in satisfaying_vacants_b):
                array_of_good_vacants = np.where(satisfaying_vacants_b == True)
                move_to = index_vacants[array_of_good_vacants[0][0]]
                self.M[random_index[0]][random_index[1]] = -1
                self.M[move_to[0]][move_to[1]] = 1
            if( True not in satisfaying_vacants_b):
                self.blocks_b = True
        if(self.blocks_a == True):
            "a agents are blocked"
            index_test = cordenates_b
            if(np.size(index_test,axis=0) != 0):
                cordenate_test = index_test[0]
                b_neights_vacants = b_neights[Y,X]
                neights_vacants = neights[Y,X]
                satisfaying_vacants_b = (b_neights_vacants >= self.sim_t*neights_vacants)
                if(True in satisfaying_vacants_b):
                    array_of_good_vacants = np.where(satisfaying_vacants_b == True)
                    move_to = index_vacants[array_of_good_vacants[0][0]]
                    self.M[cordenate_test[0]][cordenate_test[1]] = -1
                    self.M[move_to[0]][move_to[1]] = 1
                if( True not in satisfaying_vacants_b):
                    self.blocks_b = True
        if(self.blocks_b == True):
            "b agents are blocked"
            index_test = cordenates_a
            if(np.size(index_test,axis=0) != 0):
                cordenate_test = index_test[0]
                a_neights_vacants = a_neights[Y,X]
                neights_vacants = neights[Y,X]
                satisfaying_vacants_a = (a_neights_vacants >= self.sim_t*neights_vacants)
                if(True in satisfaying_vacants_a):
                    array_of_good_vacants = np.where(satisfaying_vacants_a == True)
                    move_to = index_vacants[array_of_good_vacants[0][0]]
                    self.M[cordenate_test[0]][cordenate_test[1]] = -1
                    self.M[move_to[0]][move_to[1]] = 1
                if( True not in satisfaying_vacants_a):
                    self.blocks_a = True
        if(self.blocks_a == True and self.blocks_b == True):
            self.bloked= True
        
        return self
    
    def get_matrix(self):

        return self.M
    
    def get_mean_similarity_ratio(self):
        return self._ComonOperatios.get_mean_similarity_ratio(self.M,self.empty,self.boundary)
    
    def get_mean_dissatisfaction(self):
        return self._ComonOperatios.get_mean_dissatisfaction(self.M,self.empty,self.boundary)
    
    def get_mean_interratial_pears(self):
        return self._ComonOperatios.mean_interratial_pears(self.M,self.boundary)
    
    def print_matrix(self):
        print(self.M)
