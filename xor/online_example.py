
"""
 Example of XOR gate using an MLP and MLP class
 Author: Ithallo Junior Alves Guimaraes
 Apr/2018
 Reference: http://iamtrask.github.io/2015/07/12/basic-python-network/
"""

import numpy as np
import datetime
import sys
sys.path.append("../")

from mlp import mlp

#The example
if __name__=="__main__":
    clf = mlp(seed=1, activation="tanh", max_iter=10000,
          hidden_layer_size=4, alpha=0.1, momentum=0.9,tol=1e-3, weight_range=(-1,1), bias=True)
    clf.example_run(show="online")
