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
import os

from sklearn.decomposition import PCA


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

# Viterbi training
def evaluate(model, data, w):
    i = w
    scores = []
    while(i <= data.shape[0]): 
        # finestra all'istante t-esimo della ts data[0:t]->data[1:t+1]->data[2:t+2]
        Wt = data[i-w:i].copy()
        
        # utilizza l'algoritmo di Viterbi per trovare il Viterbi path (la sequenza di stati pi`u probabile) basandosi sul modello corrente
        # ll: log probabilità della seq di stati prodotta
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
path_data = './datasets/ALFA_INTERPOLATED/'
w = 100
K = 5
###############################################

##########DATA AND MODEL TRAINING##############

dataset = []
for file in os.listdir(path_data):
    dataset.append(pd.read_csv(path_data + file).values)


# viene stabilito il modello
model = hmm.GaussianHMM(n_components=K, covariance_type="diag", n_iter=100, random_state=0)

########### NOSTRA PARTE

for i in range(len(dataset)):
    dataset[i] = np.delete(dataset[i], np.s_[0], axis=1) #rimuoviamo la colonna indice
    dataset[i] = np.delete(dataset[i], np.s_[6], axis=1)
    
    # cambio di tipo perchè la funzione np.cov da problemi di riconoscimento di tipo
    dataset[i] = dataset[i].astype('float32')


# %%
# facciamo il plot dei dati per vedere le componenti delle serie temporali

'''
index = 0
for column in training_set.transpose():
    plt.figure()
    plt.plot(column)
    plt.title("Column:" + str(index))
    plt.show()
    index += 1
#'''

# i dati vanno normalizzati rimuovendo il rumore

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


dataset_normalized = []
for d in dataset:

    dataset_normalized.append([])
    index_normaliz = len(dataset_normalized) - 1

    for column in d.transpose():
        dataset_normalized[index_normaliz].append(moving_average(column, 32))
    
    dataset_normalized[index_normaliz] = np.array(dataset_normalized[index_normaliz])
    dataset_normalized[index_normaliz] = dataset_normalized[index_normaliz].transpose()


final_dataset = []
for d in dataset_normalized:
    
    pca = PCA(n_components=5)
    final_dataset.append(pca.fit_transform(d))
    
# %%
'''
index = 0
for i in range(dataset[0].transpose().shape[0]):
    plt.figure()
    plt.plot(dataset[0].transpose()[i], 'b')
    plt.plot(dataset_normalized[0].transpose()[i], 'r')
    plt.title("Column:" + str(index))
    plt.show()
    index += 1'''


'''
cut_index = int(len(dataset)*0.9)
training_set = dataset[0:cut_index]
test_set = dataset[cut_index:]'''

########### FINE NOSTRA PARTE

# %%

'''
l = [w] * int(len(dataset_normalized)/w)
l.append(len(dataset_normalized) % int(len(dataset_normalized)/w))'''

dataset_norm_unique = []
for d in dataset_normalized:
    for row in d:
        dataset_norm_unique.append(row)

dataset_norm_unique = np.array(dataset_norm_unique)

model.fit(final_dataset[0])

###############################################

##########EVALUATION OF ANOMALY SCORE##########

anomaly_scores = evaluate(model, final_dataset[0], w)

plt.figure()
plt.plot(anomaly_scores)
plt.ylim(bottom = -0.05, top = 1.05)
plt.title("Anomaly Score on dataset")
plt.show()
###############################################