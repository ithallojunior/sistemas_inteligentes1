
# Example of XOR gate using an MLP and MLP class

## Author: Ithallo Junior Alves Guimarães

### Apr/2018

#####  Reference:     <http://iamtrask.github.io/2015/07/12/basic-python-network/>
 


```python
import numpy as np
import datetime
```


```python
class mlp():
    def __init__(self, hidden_layer_size=3, activation="sigmoid", alpha=0.1, max_iter=1000, bias=True,  
                 tol=1e-10, seed=None, keep_error_list=True, warm_start=False, coefs=None):
        self.hidden_layer_size = hidden_layer_size
        self.activation = activation
        self.alpha = alpha
        self.max_iter = max_iter
        self.bias = bias
        self.tol = tol
        self.seed = seed
        self.keep_error_list = keep_error_list
        self.error_list = []
        self.warm_start = warm_start
        self.coefs = coefs
        self.error = 0
        self.X = np.array([[0, 0],[0, 1], [1, 0], [1, 1]])
        self.y = np.array([0, 1, 1, 0])

    # activaton function
    def _function(self, X, deriv=False):
        if self.activation=="tanh":
            if(deriv==True):
                return 1. - np.square(np.tanh(X))
            else: return np.tanh(X)
        elif self.activation=="sigmoid":
            if(deriv==True):
                return X * (1 - X ) 
            else:  return 1./(1+np.exp( -X ))

        
    
    # corrects the  shape of y
    def _y_shape_corrector(self, y):
            if type(y) == list:
                    self._y_shape_corrector(np.array(y))
            else:
                if len(y.shape)==1:
                    return y.reshape(-1,1)
                else:
                    return y
    
    # creates the internal structure of the network
    def _network_constructor(self, X_, y_):
        
        #checks if there is a warm start
        if self.warm_start and (self.coefs != None):
            self.coefs = coefs
        else:
            
            # inicialize with a deterministic seed is a good practice
            np.random.seed(self.seed)
            
            #bias
            if self.bias:
                l0 = X_.shape[-1] + 1
            else:
                l0 = X_.shape[-1]
                
            #cosntructing layers with weights randomly with mean 0
            hidden_layers = 2*np.random.random((l0, self.hidden_layer_size)) - 1.
            output_layer = 2*np.random.random((self.hidden_layer_size, y_.shape[-1])) - 1.
 
            self.coefs = [hidden_layers, output_layer]
    # run
    def fit(self, X, y):
        
        #cleaning error list 
        self.error_list = []
        
        #correcting the shape of y and adding bias to X
        y = self._y_shape_corrector(y)
        
        #calling the constructor, must be after shape correction
        self._network_constructor(X, y)
        #print self.coefs
        
        #bias in the network, if needed and creating the layer 0
        if self.bias:
            layer0 = np.hstack((X, np.ones((X.shape[0], 1))))
        else:
            layer0 = X
       
        st = datetime.datetime.now()
        print "Starting MLP at:", st
        for i in xrange(self.max_iter):
            
            #training
            self._train(layer0, y)
            
            #error list, if any
            if self.keep_error_list:
                self.error_list.append(self.error)
            
            # stopping by error
            if self.error < self.tol:
                print "stopping by error"
                break
            
        ed = datetime.datetime.now()
        print "Finishing MLP training at:", ed
        print "Final error:", self.error
        print "It took %s"%(ed - st) 
        
    #trains, already receives the layer0, later maybe will add more training functions
    def _train(self, layer0, y):
        #backprogation
        
        #calculating first layer
        layer1 = self._function(np.dot(layer0, self.coefs[0]))
        #calculating for the hidden
        layer2 = self._function(np.dot(layer1, self.coefs[1]))

        #error to the target value
        l2_error = y - layer2  
        
        #slope for hidden
        l2_delta =  l2_error * self._function(layer2, deriv=True)
        
        #contribuiton to the second from the first
        l1_error = np.dot(l2_delta, self.coefs[1].T)
        
        # slope of the sigmoid at the values in layer 1
        l1_delta =  l1_error * self._function(layer1, deriv=True)
    
        
        ##updating all weights and setting the error as mean squared error
        self.coefs[1] = self.coefs[1] + self.alpha * np.dot(layer1.T, l2_delta)
        self.coefs[0] = self.coefs[0] + self.alpha * np.dot(layer0.T, l1_delta)
        self.error = np.square(l2_error).sum()/l2_error.shape[0]
    
    #predicts output after fitting a model, supposes the coefs are known    
    def predict(self, X):
        
        #bias in the network, if needed
        if self.bias:
            layer0 = np.hstack((X, np.ones((X.shape[0], 1))))
        else:
            layer0 = X
        
        layer1 = self._function(np.dot(layer0, self.coefs[0]))
        
        layer2 = self._function(np.dot(layer1, self.coefs[1]))      
        
        return layer2
    
    #calculates the score using 1 - mean square error
    def score(self, X, y):     
        #correcting the shape of y
        y = self._y_shape_corrector(y)
        y_ = self.predict(X)
        mse = np.square(y - y_).sum()/float(y.shape[0])
        return float(1. - mse)
    
    #Showing it working, example_run
    def example_run(self, plot=True):
        self.fit(self.X, self.y)
        print"Results:\n"
        print " X    y    Predicted"
        results = clf.predict(self.X)
        for i in xrange(4):
            print self.X[i],self.y[i], results[:,0][i]
        print"\nscore: %.3f%%"%(self.score(self.X,self.y)*100)
        
        if plot:
            import matplotlib.pyplot as plt
            plt.plot(self.error_list)
            plt.title("Mean Squared Error (MSE) per generation")
            plt.xlabel("Generation")
            plt.ylabel("MSE")
            plt.show()
```

## Examples


#### Using sigmoid


```python
clf = mlp(seed=1, activation="sigmoid", max_iter=300, hidden_layer_size=4, alpha=5)
%time clf.example_run()   
```

    Starting MLP at: 2018-04-15 23:28:57.799237
    Finishing MLP training at: 2018-04-15 23:28:57.826877
    Final error: 0.000949241885288
    It took 0:00:00.027640
    Results:
    
     X    y    Predicted
    [0 0] 0 0.0248102502396
    [0 1] 1 0.967929429577
    [1 0] 1 0.973253685396
    [1 1] 0 0.037640865398
    
    score: 99.906%



![png](output_5_1.png)


    CPU times: user 445 ms, sys: 74.2 ms, total: 519 ms
    Wall time: 605 ms


#### Using tanh


```python
clf = mlp(seed=1, activation="tanh", max_iter=300, hidden_layer_size=4, alpha=.4)
%time clf.example_run()   
```

    Starting MLP at: 2018-04-15 23:28:58.411577
    Finishing MLP training at: 2018-04-15 23:28:58.440488
    Final error: 0.00363558474058
    It took 0:00:00.028911
    Results:
    
     X    y    Predicted
    [0 0] 0 0.089678460178
    [0 1] 1 0.989339313261
    [1 0] 1 0.993289624639
    [1 1] 0 0.0980044181149
    
    score: 99.555%



![png](output_7_1.png)


    CPU times: user 226 ms, sys: 10.4 ms, total: 236 ms
    Wall time: 242 ms

