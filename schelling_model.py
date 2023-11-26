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

class AltruisticSchellingModel:

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
        positions2 = (convolve(positions == type,self.Kernel,**Kws))[1:4,1:4]
        return positions2

    def _calc_all_neights(self,positions,boundary='wrap'):
        Kws = dict(mode='same',boundary=boundary)
        positions = positions.reshape(5,5)
        positions2 = (convolve(positions != -1,self.Kernel,**Kws))[1:4,1:4]
        return positions2
    
    def _change_pos_type(self,positions):
        positions = positions.reshape(5,5)[1:4,1:4]
        return positions
    
    def _calc_type_dissatisfyed(self,positions):
        value = False
        if(True in positions):
            value = True
        return value
    
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
        return  M.reshape(int(self.n*self.n),int(self.n*self.n))
    
    def _check_happines_neighborhod(self,M,new_positions,type,old_position,a_neights,b_neights,neights,boundary='wrap'):
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
    
    def step_altruistic_model(self,M,bloked,blocks,boundary='wrap'):
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
        a_neights = convolve(M == 0,self.Kernel,**Kws)
        b_neights = convolve(M == 1,self.Kernel,**Kws)
        neights = convolve(M != -1,self.Kernel,**Kws)
        a_dissatisfaction = (a_neights < self.sim_t*neights)&(M == 0)
        b_dissatisfaction = (b_neights < self.sim_t*neights)&(M == 1)
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
            dissatisfaied_vacant = self._check_happines_neighborhod(M,index_vacants,agent_tipe,random_index,a_neights,b_neights,neights)
            a_neights_vacants = a_neights[Y,X]
            neights_vacants = neights[Y,X]
            satisfaying_vacants_a = (a_neights_vacants >= self.sim_t*neights_vacants)
            satisfaying_vacants_a = (satisfaying_vacants_a == True)&(dissatisfaied_vacant == False)
            if(True in satisfaying_vacants_a):
                array_of_good_vacants = np.where(satisfaying_vacants_a == True)
                move_to = index_vacants[array_of_good_vacants[0][0]]
                M[random_index[0]][random_index[1]] = -1
                M[move_to[0]][move_to[1]] = 0
            if( True not in satisfaying_vacants_a):
                blocks[0] = True
        else:
            dissatisfaied_vacant = self._check_happines_neighborhod(M,index_vacants,agent_tipe,random_index,a_neights,b_neights,neights)
            b_neights_vacants = b_neights[Y,X]
            neights_vacants = neights[Y,X]
            satisfaying_vacants_b = (b_neights_vacants >= self.sim_t*neights_vacants)
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
                dissatisfaied_vacant = self._check_happines_neighborhod(M,index_vacants,agent_tipe,random_index,a_neights,b_neights,neights)
                satisfaying_vacants_b = (b_neights_vacants >= self.sim_t*neights_vacants)
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
                dissatisfaied_vacant = self._check_happines_neighborhod(M,index_vacants,agent_tipe,random_index,a_neights,b_neights,neights)
                satisfaying_vacants_a = (a_neights_vacants >= self.sim_t*neights_vacants)
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

class ClassicSchellingModel:

    def __init__(self,N,sim_t,empty,ratio,kernel,epsilon):
        
        self.n = N
        self.sim_t = sim_t
        self.empty = empty
        self.ratio = ratio
        self.kernel = kernel
        self.epsilon = epsilon
        self.blocked = False
        self.blocks_a = False
        self.blocks_b = False

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
        return  M.reshape(int(self.n*self.n),int(self.n*self.n))
    
    def evolve(self,M,bloked,blocks_a,blocks_b,boundary='wrap'):
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
        a_neights = convolve(M == 0,self.Kernel,**Kws)
        b_neights = convolve(M == 1,self.Kernel,**Kws)
        neights = convolve(M != -1,self.Kernel,**Kws)
        a_dissatisfaction = (a_neights < self.sim_t*neights)&(M == 0)
        b_dissatisfaction = (b_neights < self.sim_t*neights)&(M == 1)
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
            satisfaying_vacants_a = (a_neights_vacants >= self.sim_t*neights_vacants)
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
            satisfaying_vacants_b = (b_neights_vacants >= self.sim_t*neights_vacants)
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
                satisfaying_vacants_b = (b_neights_vacants >= self.sim_t*neights_vacants)
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
                satisfaying_vacants_a = (a_neights_vacants >= self.sim_t*neights_vacants)
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