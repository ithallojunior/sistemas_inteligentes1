# -*- coding: utf-8 -*-
import numpy as np
import sys
sys.path.append("../")
from mlp import mlp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
import itertools
import datetime
import pdb
import os


def helper():
    os.system("clear")
    
    print "--> Seed:"
    print"Integer to mark your results and allow others to repeat it (e.g., 1, 10, None).\n"
    
    print "--> Activation function:"
    print "It must be sigmoid, tanh or softplus. Others will be considered typos.\n"
    
    print "--> Weight range:"
    print "It must follow the structure (min, max), otherwise it will be considered a typo.\n"

    print "--> Learning rate:"
    print "Number, usually less than 1. (e.g., 0.1, 0.01, 0.001).\n"
    
    print "--> Momentum:" 
    print "Number between 0. (not using it) to 1.\n"
    
    print "--> Bias:"  
    print "True or False. It must be capitalized, otherwise will be considered a typo.\n"
    
    print "--> Maximum iterations:"
    print "Number of maximum rounds to run, usually a large number (e.g., 100, 1000).\n"
    
    print "--> Minimum error:"
    print "Stopping criterion, usually less than 1. (e.g., 0.0001).\n"
    
    print "--> Hidden layer size:"
    print "Number of units in the hidden layer, greater than 0 (e.g., 3, 10, 20).\n"
    
    print "Typos will be bypassed. Remember to only use white spaces as separators\n"
    
    raw_input("Press enter to continue...")
    os.system("clear")

# plots the confusion matrix
def plot_confusion_matrix(y_test, y_pred):
    
    #pdb.set_trace()
    if raw_input("\nDo you want to plot the confusion matrix? (y/N): ")=="y":
    
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
        plt.show()
        plt.close('all')
        plt.clf()
        



# receives data
def data_input(mydict):
    tanh = "tanh"
    sigmoid = "sigmoid"
    softplus = "softplus"
    mykeys = mydict.keys()
    sz = len(mykeys)
    print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    print "# | Variable  |  Value\n"
    for i in xrange(sz):
        print "%s | %s: %s"%(i, mykeys[i], mydict[mykeys[i]])
    print "\nType help for, you guessed right, help."
    print "Use only white spaces to separate."
    print "Wrong numbers or empty will call the previous/default ones."
    to_change = str(raw_input("Type the numbers of variables to be changed:\n")).split()

    #print to_change
    if to_change!=[]:
        #calling helper
        if "help" in to_change:
            helper()
        
    
    for i in to_change:
        try:
            j = int(i)
            if (j>=0) and (j<sz):
                mydict[mykeys[j]] = eval(raw_input("Type value for %s: "%(mykeys[j])))
        except ValueError:
            #print "Wrong value, values set to default/previous \n"
            pass
    #print mydict
    print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    return mydict


# plot error on the go
def error_plotter(error_list):
    if raw_input("\nDo you want to plot the error on the go? (y/N): ")=="y":
        #pdb.set_trace()
        fig = plt.figure(1) 
        ax = plt.axes(xlim=(0, len(error_list)), ylim=(0, max(error_list)))
        xdata, ydata = [], []
        ln, = ax.plot([], [], lw=2)

        def init():
            ln.set_data([],[])
            return ln,

        def update(frame):
            xdata.append(frame)
            ydata.append(error_list[frame])
            ln.set_data(xdata, ydata)
            return ln,

        plt.title("Squared Error per iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Squared Error")

        ani = FuncAnimation(fig, update, frames=range(len(error_list)),
            init_func=init, blit=True,interval=10)
        plt.show()
        plt.close('all')
        plt.clf()
        
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


    param = {"Seed":1277,
        "Activation function":"tanh",
        "Maximum iterations":1000, 
        "Hidden layer size":12, 
        "Learning rate":0.001, 
        "Momentum":0.9, 
        "Maximum error":1e-3, 
        "Weight range":(-1., 1.),
        "Bias":True}
    #loop
    while(1):     
        try:
            print "Type ctrl-c to exit"
            #pdb.set_trace()
            param = data_input(param)
            #print param, "\n\n\n\n\n"

            clf = mlp(seed=param["Seed"], activation=param["Activation function"], max_iter=param["Maximum iterations"], 
                hidden_layer_size=param["Hidden layer size"], alpha=param["Learning rate"], momentum=param["Momentum"], 
                tol=param["Maximum error"], weight_range=param["Weight range"], bias=param["Bias"], classifier=True)
            
            clf.fit(X_train, y_train)

            print "\nValidation score: %.2f%%"%(clf.score(X_validation, y_validation) *100)
            
            ##################################################################################################
            
            if 'y'!=raw_input("\n\nDo you want to change anything and run again for validation? (y/N): "):
                    
                #plotting the error
                error_plotter(clf.error_list) 
                
                #print test score
                #pdb.set_trace()
                
                
                names = data.name[myray[int(0.7*X_data.shape[0]):int(0.85*X_data.shape[0])]].values
                pred = clf.predict(X_test)[:,0]
                 
                #plotting confusion matrix
                plot_confusion_matrix(y_test, pred)
               
                os.system("clear")
                print "Final Test Score "
                print "Test score: %.2f%%\n"%(clf.score(X_test, y_test) *100) 
                print "Patient label  |    Result    |  Expected\n"
                for i in xrange(y_test.shape[0]):
                    result = np.where(pred==1., "   Parkinson", "No Parkinson")
                    ex = np.where(y_test==1., "Parkinson", "No Parkinson")
                    print ("%s : %s | %s")%(names[i], result[i], ex[i])
                
                
                break
        except (TypeError, NameError, SyntaxError) as e:
            os.system("clear")
            print "\n\nYou had a typo, please repeat.\n\n"
            raw_input("Press enter to continue...")
            os.system("clear")
        except KeyboardInterrupt:
            os.system("clear")
            break
    
        os.system("clear")
    
    print "\nFinished"


def welcome():
    try: 
        os.system("clear")
        print "             #######################################"
        print "             ##                                   ##" 
        print "             ##  This software aims to predict    ##" 
        print "             ##     whether the patient has       ##" 
        print "             ##  Parkinson disease or not based   ##" 
        print "             ##  on the voice and some features   ##" 
        print "             ##  of it.                           ##" 
        print "             ##                                   ##" 
        print "             #######################################"

        print "\n\nDisclaimer:"
        print "This program is just a proof"
        print "of concept and in ABSOLUTE NO WAY should be" 
        print "used for a diagnostics purpose." 
        print "Use it at your own risk."

        print "\n\nDeveloped by:"
        print "Ithallo J.A.G.,José E.S.," 
        print "Renata M.L. and Roberta M.L.C."

        print "\n\nDataset obtained from: https://archive.ics.uci.edu/ml/datasets/Parkinsons"
        print "Contact ithallojunior@outlook.com for more information."
        raw_input("Press enter to continue...")
        os.system("clear")
    
    except KeyboardInterrupt:
        print "\nExiting \n" 
        os._exit(1)
if __name__=="__main__":
        welcome()
        interface()
