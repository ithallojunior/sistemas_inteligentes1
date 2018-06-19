import numpy as np
import sys
sys.path.append("../")
from mlp import mlp
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import itertools
import datetime
import pickle

## confusion matrix
def cm(y_test, y_pred, to_file=True):
    cnf_matrix = confusion_matrix(y_test, y_pred)
    classes = np.array(["No Parkinson", "Parkinson"])
    plt.clf()
    plt.close("all")
    #plt.figure(figsize = (7.5,6))
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    #normalized
    cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, round(cnf_matrix[i, j], 3),
            horizontalalignment="center",
            color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Expected label')
    plt.xlabel('Predicted label')
    if to_file:
        plt.savefig("my.png")
        plt.close()
    else:
        plt.show()


#data reader, specific for the problem, returns the X and y data, not scaled
#also makes the number of input even between classes
def specific_data_reader(x_list=[22, 19, 0, 1, 2], to_scale=True):
    data = pd.read_csv("../dataset/parkinsons.data")
    #print "total PD", data.status.values.tolist().count(1)
    #print "total no PD", data.status.values.tolist().count(0)
    # as the number of PD and no PD is hugely different, resampling to get even amounts
    pd_index = data.index[data.loc[:, "status"]==1]
    nopd_index = data.index[data.loc[:, "status"]==0]
    #print pd_index.shape
    #print nopd_index.shape
    
    #making data more even
    my_index = np.hstack((pd_index[:48], nopd_index))
    #print my_index.shape
    
    #actual data
    X = data.values[my_index, 1:].astype(np.float64)
    y = data.status.values.astype(np.float64)[my_index]#.reshape(-1,1)
    if to_scale:
        scaler = MinMaxScaler()    
        scaler.fit(X)
        X = scaler.transform(X) #data[data.columns[[1, 2, 3]]].values
        X_sc = X[:, x_list]
        #print X.shape, y.shape

        return X_sc,y       
    return X,y
 
    
## genetic algorithm, generic enough to any amount of input data with 2 classes
class genetic_population_creator():

    def __init__(self, mlp, example_population, population_size=100,
                 class_a_to_b_prop=0.5, crossover_rate=0.5, mutate_rate=0.1, 
                 total_generations=1000, verbose=False, seed=None, error_stop=1e-3):
        
        self.mlp = mlp # the instace of the MLP already traine
        self.population = None
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.total_generations = total_generations
        self.verbose = verbose
        self.seed = seed
        self.error_stop = error_stop
        self.errorA = []
        self.errorB = []
        
        #general purpose variables
        self.my_max = np.amax(example_population, axis=0)
        self.my_min = np.amin(example_population, axis=0)         

        self._rank = None
        self.fitnesses = None
        self._fitness_array = None
        self.to_mutate_n = int(population_size * mutate_rate)
        self.to_crossover_n = int(crossover_rate * population_size)
        self.inv_to_crossover_n = int(self.population_size - self.to_crossover_n)
        self.class0_number = int(class_a_to_b_prop * population_size)
        
        
    #creates the population    
    def _create_population(self):
        np.random.seed(self.seed)
        #print my_max, my_min
        pop = [] #CHEAT BELOW
        for mn,mx  in zip(self.my_min, self.my_max):
            if (mn<0) and (mx<0):
                 pop.append(-(np.random.normal(-mx, -mn, (self.population_size))))
            else:
                pop.append(np.random.normal(mn, mx, (self.population_size)))
        
        self.population = np.abs(np.array(pop).T).round(3) #TODO REDO
     
    
    #calculates the fitness and ranks the population, the closer to zero, the better
    # returns the rank (numbers from best to worst) and the error list sorted ()
    def _get_fitnesses(self, myclass=0):

        prediction = self.mlp.predict(self.population)[:,0]

        #as we want to set the proportions, there are two fitnesses in one
        # half the pop takes one, the other takes the second
        
        if myclass==0:
            rank_not_sorted = np.abs(prediction) # the closer to 0, the better
        else:
             rank_not_sorted = np.abs(1. - prediction)

        self._rank =  rank_not_sorted.argsort()
        self._fitness_array = np.sort(rank_not_sorted)
    
    #calculates the external final fitness, assumes class a=>0, class b=>1 and resorts the population
    def _final_sorter_grouper(self, class0, class1):  
        pred_class0 = self.mlp.predict(class0)[:,0]
        pred_class1 = self.mlp.predict(class1)[:,0]         
        
        rank_not_sorted_class0 = np.abs(pred_class0)
        rank_not_sorted_class1 = np.abs(1. - pred_class1)
        #print rank_not_sorted_class0.shape, rank_not_sorted_class1.shape
        
        rank_not_sorted = np.hstack((rank_not_sorted_class0[:self.class0_number], 
            rank_not_sorted_class1[self.class0_number:]))
          
        self.population = np.vstack((class0[:self.class0_number, :], class1[self.class0_number:, :]))
        self.population = self.population[rank_not_sorted.argsort()]
        self.fitnesses = np.sort(rank_not_sorted)
        
    # makes the crossover and removes the less fitted (the number of indivuals always stays the same)
    def _crossover(self):
        
        np.random.seed(self.seed)
        
        #print"cross", n_cross, "inv cross" ,inv_n_cross 
        
        # aligning the population to the order of the rank
        self.population = self.population[self._rank]
        #print "\n pop", self.population
        
        #other from all array, to maintain the diversity
        other_elements = np.copy(self._rank)
        np.random.shuffle(other_elements)
        #print "\n shuffle\n", other_elements 
        
        #choosing randomly th point to cross, as it is an odd number
        cross_point = np.random.randint(1, self.population.shape[1])
        #print "\n cross point", cross_point
        
        #crossing the bests with the others and removing excessive data
        rand_pop = self.population[other_elements]
        data = np.hstack((self.population[:, :cross_point], rand_pop[:, cross_point:]))
        #print "\n data\n", data
        
        #adding new individuals to end and removing the worst ones
        #print"cross", n_cross, "inv cross" ,inv_n_cross
        
        old_pop = self.population[:self.inv_to_crossover_n, :]
        new_pop = data[:self.to_crossover_n, :]
        #print"old", old_pop.shape,"\n new", new_pop.shape 

        self.population = np.vstack((old_pop, new_pop))
        
        #print "\n pop\n", self.population
    
    # mutates the remaining population
    def _mutate(self):
                
        # no reason to run if not to mutate
        if self.to_mutate_n > 0:
            np.random.seed(self.seed)
            my_array = np.copy(self._rank)
            np.random.shuffle(my_array)
            #indivuals to mutate
            mutated_individuals = my_array[:self.to_mutate_n]
            #mutated cromossomes
            mutated_cromossomes = np.random.randint(0, self.population.shape[1], (self.to_mutate_n))
            #print"\n muatation" 
            #print mutated_individuals, mutated_cromossomes
            pop = []
            for mn,mx  in zip(self.my_min, self.my_max): #CHEAT
                if (mn<0) and (mx<0):
                    pop.append(-(np.random.normal(-mx, -mn, (self.to_mutate_n))))
                else:
                    pop.append(np.random.normal(mn, mx, (self.to_mutate_n)))
        
            m_pop = np.abs(np.array(pop).T).round(3)[:, mutated_cromossomes][:, 0] #TODO REDO
            #print m_pop
        
            self.population[mutated_individuals, mutated_cromossomes] = m_pop
        
    
    #receives noinput and returns the generated population
    def fit(self):
        
        # generating class A
        
        if self.verbose:
            print "Finding class A"
        
        self._create_population()  
        self.errorA = []
        for i in xrange(self.total_generations):
            
            self._get_fitnesses(myclass=0)
            
            my_tot_error = self._fitness_array.sum()
            
            self.errorA.append(my_tot_error)
            
            if self.verbose:
                print "---> Summed scores: %s"%self._fitness_array.sum()
            if my_tot_error<=self.error_stop:
                print "A stopping by error at %s"%i
                break
            
            self._crossover()
            self._mutate()
        class0 = np.copy(self.population) 
        
        # generating class B  
        if self.verbose:
            print "Finding class B"
            
        self._create_population()
        self.errorB = []
        for i in xrange(self.total_generations):
            
            self._get_fitnesses(myclass=1)
            
            my_tot_error = self._fitness_array.sum()
            
            self.errorB.append(my_tot_error)
            
            if self.verbose:
                print "---> Summed scores: %s"%self._fitness_array.sum()
            if my_tot_error<=self.error_stop:
                print "B stopping by error at %s"%i
                break          
            
            self._crossover()
            self._mutate()
            
        # setting the final rank joining  and the population back together
        self._final_sorter_grouper(class0, self.population)
                
        
        
if __name__=="__main__":
    
    X, y = specific_data_reader()
    
    f = open("pd.mlp", "r")
    clf = pickle.loads(f.read())
    f.close()
    
    ga = genetic_population_creator(clf, X, population_size=10, seed=0,
        verbose=False, total_generations=100, error_stop=1-10,
        crossover_rate=0.5, mutate_rate=0.2, class_a_to_b_prop=0.5)
    
    ga.fit()

    print "Final Population"
    print ga.population
    x = clf.predict(ga.population)
    print "Classes:"
    print np.where(1, x>=0.5, 0)[:,0]
    
    #plotting
    plt.subplot(211)
    plt.plot(ga.errorA)
    plt.title("Error A0")
    plt.subplot(212)
    plt.plot(ga.errorB)
plt.title("Error B1")
plt.show()
    
