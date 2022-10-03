
import numpy as np
from scipy.signal import convolve2d as convolve
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.colors import ListedColormap






class Schelling:
    def __init__(self,size,empty_ratio,similarity_threshold,etnic_ratio,boundary='wrap') -> None:
        self.size = size
        self.empty_ratio = empty_ratio
        self.similarity_threshold = similarity_threshold
        self.etnic_ratio = etnic_ratio
        self.boundary = boundary
        self.Kernel = np.array([[1,1,1],[1,0,1],[1,1,1]],dtype=np.int8)
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
        count = 0
        similarity_ratio = 0
        for (row, col), value in np.ndenumerate(self.city):
            race = self.city[row, col]
            if race != 0:
                neighborhood = self.city[row-1:row+1, col-1:col+1]
                neighborhood_size = np.size(neighborhood)
                n_empty_houses = len(np.where(neighborhood == -1)[0])
                if neighborhood_size != n_empty_houses + 1:
                    n_similar = len(np.where(neighborhood == race)[0]) - 1
                    similarity_ratio += n_similar / (neighborhood_size - n_empty_houses - 1.)
                    count += 1
        return similarity_ratio / count



    


st.title("Schelling's Model of Segregation")
#population_size = st.sidebar.slider("Population Size", 500, 10000, 2500)
empty_ratio = st.sidebar.slider("Empty Houses Ratio", 0., 1., .2)
similarity_threshold = st.sidebar.slider("Similarity Threshold", 0., 1., .4)
n_iterations = st.sidebar.number_input("Number of Iterations", 50)

schelling = Schelling(60, empty_ratio, similarity_threshold, 1)
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
plt.ylim([0.4, 1])
plt.title("Mean Similarity Ratio", fontsize=15)
plt.text(1, 0.95, "Similarity Ratio: %.4f" % schelling.get_mean_similarity_ratio(), fontsize=10)

city_plot = st.pyplot(plt)

progress_bar = st.progress(0)

if st.sidebar.button('Run Simulation'):

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
        plt.ylim([0.4, 1])
        plt.title("Mean Similarity Ratio", fontsize=15)
        plt.plot(range(1, len(mean_similarity_ratio)+1), mean_similarity_ratio)
        plt.text(1, 0.95, "Similarity Ratio: %.4f" % schelling.get_mean_similarity_ratio(), fontsize=10)

        city_plot.pyplot(plt)
        plt.close("all")
        progress_bar.progress((i+1.)/n_iterations)


