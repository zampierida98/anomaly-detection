# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 11:41:00 2021

@author: Francesco Masillo
"""
import pandas as pd
import numpy as np
from scipy import stats, linalg
from hmmlearn import hmm
import matplotlib.pyplot as plt


def hellinger_dist(mu_model, cov_model, mu_data, cov_data):
    num_comp1 = (linalg.det(cov_model)**(1/4))*(linalg.det(cov_data)**(1/4))
    den_comp1 = (linalg.det((cov_model + cov_data)/2)**(1/2))
    comp1 = num_comp1/den_comp1
    comp2 = float(np.exp((-1/8) * (mu_model - mu_data) @ np.linalg.matrix_power((cov_model+cov_data)/2, -1) @ (mu_model - mu_data).T))
    return 1 - comp1 * comp2

def evaluate(model, data, w):
    i = w
    scores = []
    while(i <= data.shape[0]): 
        Wt = data[i-w:i].copy()
        ll, St = model.decode(Wt)
        st = stats.mode(St)[0]
        X = Wt[St == st]
        mu = np.reshape(np.mean(X, axis=0), [1, data.shape[1]])
        cov = (np.diag(np.cov(X.T)) + 1e-5) * np.eye(data.shape[1], data.shape[1])
        score = hellinger_dist(model.means_[st], model.covars_[st][0], mu, cov)
        
        scores.append(score)
        i += 1
    
    return scores

##########PARAMETERS############################
path_data = 'dataset.csv'
w = 100
K = 5
###############################################

##########DATA AND MODEL TRAINING##############
dataset = pd.read_csv(path_data).values
model = hmm.GaussianHMM(n_components=K, covariance_type="diag", n_iter=100, random_state=0)

model.fit(dataset)
###############################################

##########EVALUATION OF ANOMALY SCORE##########
anomaly_scores = evaluate(model, dataset, w)

plt.figure()
plt.plot(anomaly_scores)
plt.ylim(bottom = -0.05, top = 1.05)
plt.title("Anomaly Score on dataset")
plt.show()
###############################################