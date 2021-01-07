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


# COV_DATA: set of the probability distributions over observations in each state
# hellinger_dist: 1/2 * integral(sqrt (f(x)) - sqrt (g(x)) dx) 
def hellinger_dist(mu_model, cov_model, mu_data, cov_data):
    num_comp1 = (linalg.det(cov_model)**(1/4))*(linalg.det(cov_data)**(1/4))
    den_comp1 = (linalg.det((cov_model + cov_data)/2)**(1/2))
    comp1 = num_comp1/den_comp1
    comp2 = float(np.exp((-1/8) * (mu_model - mu_data) @ np.linalg.matrix_power((cov_model+cov_data)/2, -1) @ (mu_model - mu_data).T))
    return 1 - comp1 * comp2


# viene calcolato uno score ad ogni sliding window.
# Quando supera lo threshold è una anomalia.

# Online Anomaly Detection
def evaluate(model, data, w):
    i = w
    scores = []
    while(i <= data.shape[0]): 
        # finestra all'istante t-esimo della ts data[0:t]->data[1:t+1]->data[2:t+2]
        Wt = data[i-w:i].copy()
        
        # Find most likely state sequence corresponding to Wt
        # ll: probabilità log della seq di stati prodotta
        # St: seq di stati prodotta. Stessa len di Wt ovvimente
        ll, St = model.decode(Wt)
        
        # ritorna un array contenente i valori più comuni nell'array passato come param
        st = stats.mode(St)[0]
        
        # #t through maximum likelihood with the data inside the window
        X = Wt[St == st]

        mu = np.reshape(np.mean(X, axis=0), [1, data.shape[1]])
        cov = (np.diag(np.cov(X.T)) + 1e-5) * np.eye(data.shape[1], data.shape[1])
        
        
        # calcolo con hellinger_dist lo score 
        # Then the Hellinger distance is
        # computed between Norm(µ, Sigma) and the emission
        # probability of state sˆt in lambda^N
        
        score = hellinger_dist(model.means_[st], model.covars_[st][0], mu, cov)
        
        scores.append(score)
        i += 1
    
    return scores

##########PARAMETERS############################
path_data = './datasets/ALFA_INTERPOLATED/carbonZ_2018-07-18-15-53-31_1_engine_failure.csv'
w = 100
K = 5
###############################################

##########DATA AND MODEL TRAINING##############
dataset = pd.read_csv(path_data).values
# posizione originale
model = hmm.GaussianHMM(n_components=K, covariance_type="diag", n_iter=100, random_state=0)

########### NOSTRA PARTE
dataset = np.delete(dataset, np.s_[0], axis=1) #rimuoviamo la colonna indice
dataset = np.delete(dataset, np.s_[6], axis=1)

# cambio di tipo perchè la funzione np.cov da problemi di riconoscimento di tipo
dataset = dataset.astype('float32')
########### FINE NOSTRA PARTE


model.fit(dataset)

###############################################

##########EVALUATION OF ANOMALY SCORE##########
anomaly_scores = evaluate(model, dataset, 100)

plt.figure()
plt.plot(anomaly_scores)
plt.ylim(bottom = -0.05, top = 1.05)
plt.title("Anomaly Score on dataset")
plt.show()
###############################################