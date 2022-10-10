
import numpy as np
from scipy.signal import convolve2d as convolve
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.colors import ListedColormap






class Schelling:
    def __init__(self,size,empty_ratio,similarity_threshold,etnic_ratio,neightborhod_size,boundary='wrap') -> None:
        self.size = size
        self.empty_ratio = empty_ratio
        self.similarity_threshold = similarity_threshold
        self.etnic_ratio = etnic_ratio
        self.boundary = boundary
        kernel = np.ones((2*neightborhod_size+1,2*neightborhod_size+1),dtype=np.int8)
        kernel[neightborhod_size][neightborhod_size] =0 
        self.Kernel = kernel
        self.city = np.zeros((int(self.size),int(self.size)))

    def rand_init(self):
        """ Random grid initialitzation
        A = 0
        B = 1
        empty = -1
        """
        vacant = self.size*self.size*self.empty_ratio
        population = self.size**2-vacant
        A = int(population*1/(1+1/self.etnic_ratio))
        B = int(population-A)
        M =np.zeros(int(self.size*self.size),dtype=np.int8)
        M[:B] = 1
        M[int(-vacant):] = -1
        np.random.shuffle(M)
        self.city =  M.reshape(int(self.size),int(self.size))

    def evolve(self):
        """
        Args:
            M(numpy.array): the matrix to be evolved
            boundary(str): Either wrap, fill or symm
        if the simlarity ratio of neighbours
        to the enitre neghtbourhood polupation
        is lower than sim_t,
        then the individual moves to an empty house. 
        """
        Kws = dict(mode='same',boundary=self.boundary)
        a_neights = convolve(self.city == 0,self.Kernel,**Kws)
        b_neights = convolve(self.city == 1,self.Kernel,**Kws)
        neights = convolve(self.city != -1,self.Kernel,**Kws)
        
        a_dissatisfaction = (a_neights/neights < self.similarity_threshold)&(self.city == 0)
        b_dissatisfaction = (b_neights/neights < self.similarity_threshold)&(self.city == 1)
        self.city[b_dissatisfaction | a_dissatisfaction] = -1
        vacant = (self.city == -1).sum()

        n_a_dissatisfied, n_b_dissatisfied = a_dissatisfaction.sum(),b_dissatisfaction.sum()
        filling = -np.ones(vacant,dtype=np.int8)
        filling[:int(n_a_dissatisfied)] = 0
        filling[int(n_b_dissatisfied):int(n_b_dissatisfied+n_a_dissatisfied)] = 1
        np.random.shuffle(filling)
        self.city[self.city == -1] = filling
    
 

    
    
    def get_mean_similarity_ratio(self):

        Kws = dict(mode='same',boundary=self.boundary)
        a_neights = convolve(self.city == 0,self.Kernel,**Kws)
        b_neights = convolve(self.city == 1,self.Kernel,**Kws)
        neights = convolve(self.city != -1,self.Kernel,**Kws)
        n_similar_a = np.where(np.logical_and(self.city !=-1,self.city == 0),\
            a_neights/neights,0)
        n_similar_b = np.where(np.logical_and(self.city !=-1,self.city == 1),\
            b_neights/neights,0)
        n_similar = np.sum((n_similar_a+n_similar_b))
        return n_similar/np.size(self.city)






    


st.title("Schelling's Model of Segregation")
population_size = st.sidebar.slider("Population Size", 500, 10000, 3600)
red_blue_ratio = st.sidebar.slider("ratio from read to blue",0.1,5.,1.)
empty_ratio = st.sidebar.slider("Empty Houses Ratio", 0., 1., .2)
similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.1, 1., .4)
neightborhod_size = st.sidebar.slider("Neightborhod size",1,5,1)
n_iterations = st.sidebar.number_input("Number of Iterations", 50)

schelling = Schelling(int(np.sqrt(population_size)), empty_ratio, similarity_threshold, red_blue_ratio,neightborhod_size)
schelling.rand_init()
mean_similarity_ratio = []
mean_similarity_ratio.append(schelling.get_mean_similarity_ratio())

#Plot the graphs at initial stage
plt.style.use("ggplot")
plt.figure(figsize=(8, 4))

# Left hand side graph with Schelling simulation plot
cmap = ListedColormap(['white', 'red', 'royalblue'])
plt.subplot(121)
plt.axis('off')
plt.pcolor(schelling.city, cmap=cmap, edgecolors='w', linewidths=1)

# Right hand side graph with Mean Similarity Ratio graph
plt.subplot(122)
plt.xlabel("Iterations")
plt.xlim([0, n_iterations])
plt.ylim([0.3, 1])
plt.title("Mean Similarity Ratio", fontsize=15)
plt.text(1, 0.95, "Similarity Ratio: %.4f" % schelling.get_mean_similarity_ratio(), fontsize=10)

city_plot = st.pyplot(plt)

progress_bar = st.progress(0)
if st.sidebar.button('Run Simulation'):
    print(np.sum(schelling.city ==-1))
    for i in range(n_iterations):
        schelling.evolve()
        mean_similarity_ratio.append(schelling.get_mean_similarity_ratio())
        plt.figure(figsize=(8, 4))

        plt.subplot(121)
        plt.axis('off')
        plt.pcolor(schelling.city, cmap=cmap, edgecolors='w', linewidths=1)

        plt.subplot(122)
        plt.xlabel("Iterations")
        plt.xlim([0, n_iterations])
        plt.ylim([0.3, 1])
        plt.title("Mean Similarity Ratio", fontsize=15)
        plt.plot(range(1, len(mean_similarity_ratio)+1), mean_similarity_ratio)
        plt.text(1, 0.95, "Similarity Ratio: %.4f" % schelling.get_mean_similarity_ratio(), fontsize=10)

        city_plot.pyplot(plt)
        plt.close("all")
        progress_bar.progress((i+1.)/n_iterations)

    print(np.sum(schelling.city ==-1))

