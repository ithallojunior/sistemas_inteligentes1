import numpy as np
import sys
sys.path.append("../")
from mlp import mlp
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import scale
import datetime
import pdb
import os

def data_input(mydict):
    sigmoid = "sigmoid"
    tanh = "tanh"
    softplus = "softplus"
    mykeys = mydict.keys()
    print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    print "Variable      Default Value\n"
    for i in mykeys:
        print "%s: %s"%(i, mydict[i])
    to_change = raw_input("\nType the names of variables to be changed, leave empty to default:\n").split()

    for i in to_change:
        if i in mykeys:
            mydict[i] = eval(raw_input("Type value for %s: "%(i)))

    #print mydict
    print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    return mydict


def interface():
    
    #getting dataset
    data = pd.read_csv("dataset/parkinsons.data")
    
    #preprocessing data
    pd_index = data.index[data.loc[:, "status"]==1]
    nopd_index = data.index[data.loc[:, "status"]==0]
    #making data more even
    my_index = np.hstack((pd_index[:48], nopd_index))
    X = scale(data.values[my_index, 1:].astype(np.float64)) 
    y = data.status.values.astype(np.float64)[my_index]
    
    np.random.seed(0)#0
    X_data = X[:, [22, 19, 0, 1]] #X[:, [22, 21, 20, 19]]  #X[:, [0,1,2]]
    myray = np.arange(X_data.shape[0])
    np.random.shuffle(myray)


    X_train = X_data[myray[:int(0.7*X_data.shape[0])],:]
    X_test = X_data[myray[int(0.7*X_data.shape[0]):int(0.85*X_data.shape[0])],:]
    X_validation = X_data[myray[int(0.85*X_data.shape[0]):],:]


    y_train = y[myray[:int(0.7*X.shape[0])]]
    y_test = y[myray[int(0.7*X.shape[0]):int(0.85*X.shape[0])]]
    y_validation = y[myray[int(0.85*X.shape[0]):]]


    myparam = {"seed":1277,
        "activation":"tanh",
        "max_iter":1000, 
        "hidden_layer_size":12, 
        "alpha":0.001, 
        "momentum":0.9, 
        "tol":1e-3, 
        "weight_range":(-1., 1.),
        "bias":True}
    #loop
    while(1):
        
        try:
            print "Type ctrl-c to exit"
            #pdb.set_trace()
            param = data_input(myparam)
            #print param, "\n\n\n\n\n"

            clf = mlp(seed=param["seed"], activation=param["activation"], max_iter=param["max_iter"], 
                hidden_layer_size=param["hidden_layer_size"], alpha=param["alpha"], momentum=param["momentum"], 
                tol=param["tol"], weight_range=param["weight_range"], bias=param["bias"], classifier=True)
            
            clf.fit(X_train, y_train)

            print "Validation:%.2f"%(clf.score(X_validation, y_validation) *100)
            
            if 'y'!=raw_input("Do you want to change something and run again for validation? (y/N)\n"):
                print "Final Test"
                print "Test:%.2f\n"%(clf.score(X_test, y_test) *100) 
                names = data.name[myray[int(0.7*X_data.shape[0]):int(0.85*X_data.shape[0])]].values
                pred = clf.predict(X_test)[:,0]
                print "Pacient name   |    Result    |  Expected\n"
                for i in xrange(y_test.shape[0]):
                    result = np.where(pred==1., "   Parkinson", "No Parkinson")
                    ex = np.where(y_test==1., "Parkinson", "No Parkinson")
                    print ("%s : %s | %s")%(names[i], result[i], ex[i])
                break
        except KeyboardInterrupt:
            break

    print "\nExited"

if __name__=="__main__":
        os.system("clear")

        print "#######################################"
        print "##                                   ##" 
        print "##  This software aims to predict    ##" 
        print "##     whether the patient has       ##" 
        print "##  Parkinson disease or not based   ##" 
        print "##  on the voice and some features   ##" 
        print "##  of it.                           ##" 
        print "##                                   ##" 
        print "#######################################"
        interface()
