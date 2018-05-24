
"""
 Example of XOR gate using an MLP and MLP class
 Author: Ithallo Junior Alves Guimaraes
 Apr/2018
 Reference: http://iamtrask.github.io/2015/07/12/basic-python-network/
"""

import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from mlp import mlp

#The example
if __name__=="__main__":

    X = np.linspace(1, 10, 100).reshape(-1,1)
    y = pow(X, 3)
    y_std = y/10000.

    reg = mlp(hidden_layer_size=150, activation="sigmoid", alpha=0.1,
        max_iter=10000, bias=True, tol=1e-10, seed=None, keep_error_list=True,
        momentum=0.9, weight_range=(-1.,1))

    reg.fit(X, y_std)
    y_pred = reg.predict(X) * 10000.

    plt.subplot(211)
    plt.title("Final score: %.4f"%(r2_score(y, y_pred)))
    plt.grid()
    plt.plot(reg.error_list)
    plt.subplot(212)
    plt.plot(X, y, c="b", label="Expected")
    plt.plot(X, y_pred, c="r", label="Predicted")
    plt.grid()
    plt.legend()
    plt.show()
