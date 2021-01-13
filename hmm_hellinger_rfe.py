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

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

from statsmodels.tsa.seasonal import seasonal_decompose


# COV_DATA: set of the probability distributions over observations in each state
# hellinger_dist: 1/2 * integral(sqrt (f(x)) - sqrt (g(x)) dx) 
def hellinger_dist(mu_model, cov_model, mu_data, cov_data):
    num_comp1 = (linalg.det(cov_model)**(1/4))*(linalg.det(cov_data)**(1/4))
    den_comp1 = (linalg.det((cov_model + cov_data)/2)**(1/2))
    comp1 = num_comp1/den_comp1
    comp2 = float(np.exp((-1/8) * (mu_model - mu_data) @ np.linalg.matrix_power((cov_model+cov_data)/2, -1) @ (mu_model - mu_data).T))
    return 1 - comp1 * comp2


# viene calcolato uno score ad ogni sliding window
# quando supera la threshold è una anomalia

# Viterbi training
def evaluate(model, data, w):
    i = w
    scores = []
    while(i <= data.shape[0]): 
        # finestra all'istante t-esimo della ts data[0:t]->data[1:t+1]->data[2:t+2]
        Wt = data[i-w:i].copy()
        
        # utilizza l'algoritmo di Viterbi per trovare il Viterbi path (la sequenza di stati più probabile) basandosi sul modello corrente
        # ll: log probabilità della sequenza di stati prodotta
        # St: sequenza di stati prodotta (stessa lunghezza di Wt)
        ll, St = model.decode(Wt)
        
        # ritorna un array contenente i valori più comuni nell'array passato come parametro
        st = stats.mode(St)[0]
        
        # t through maximum likelihood with the data inside the window
        X = Wt[St == st]

        mu = np.reshape(np.mean(X, axis=0), [1, data.shape[1]])
        cov = (np.diag(np.cov(X.T)) + 1e-5) * np.eye(data.shape[1], data.shape[1])
        
        # calcolo con hellinger_dist lo score 
        # the Hellinger distance is computed between Norm(µ, Sigma) and the emission probability of state s^t in lambda^N
        score = hellinger_dist(model.means_[st], model.covars_[st][0], mu, cov)
        
        scores.append(score)
        i += 1
    
    return scores


##########PARAMETERS############################
path_data = './datasets/ALFA_INTERPOLATED/carbonZ_2018-07-18-15-53-31_1_engine_failure.csv'
path_data_2 = './datasets/ALFA_INTERPOLATED/carbonZ_2018-07-18-15-53-31_2_engine_failure.csv'
w = 100
K = 5
###############################################

##########DATA AND MODEL TRAINING##############
dataset_1 = pd.read_csv(path_data).values
dataset_2 = pd.read_csv(path_data_2).values
dataset = np.concatenate((dataset_1, dataset_2))
###############################################


def feature_extraction(dataset):
    model = hmm.GaussianHMM(n_components=K, covariance_type="diag", n_iter=100, random_state=0)
    
    dataset = np.delete(dataset, np.s_[0], axis=1)
    
    Y = dataset[:,6]
    dataset = np.delete(dataset, np.s_[6], axis=1)
    
    Y = Y.astype('float32')
    dataset = dataset.astype('float32')
    
    # https://www.datacamp.com/community/tutorials/feature-selection-python
    # Standardizing the features
    dataset = StandardScaler().fit_transform(dataset)
    # Feature extraction
    m = LogisticRegression()
    rfe = RFE(m, n_features_to_select=3)
    fit = rfe.fit(dataset, Y)
    print("Num Features: %s" % (fit.n_features_))
    print("Selected Features: %s" % (fit.support_))
    print("Feature Ranking: %s" % (fit.ranking_))
    
    feat = []
    for i in range(len(fit.ranking_)):
        if fit.ranking_[i] == 1:
            feat.append(i)
    
    
    train = dataset[0:2221,feat]
    test = dataset[2221:,feat]
    
    ########## RIDUZIONE DEL RUMORE CON ROLLING MEAN ############
    seconds = 0.5
    
    dataset_normalized = []
    for column in train.transpose():
        
        # DATI IN SCALA MA INUTILE QUI MAGARI INFLUISCE SE SPOSTATO SOPRA
        '''
        scale = max(abs(column))
        if scale != 0:
            column /= scale
        dataset_normalized.append(column)
        '''
        
        # rolling mean 
        column = pd.Series(column).rolling(int(16*seconds)).mean()
        column.dropna(inplace=True)
        
        # decomposizione serie temporale
        ts_column = pd.Series(column)
        decomposition = seasonal_decompose(ts_column, model="additive", period=200)
        
        residual_absent = decomposition.seasonal + decomposition.trend
        residual_absent.dropna(inplace=True)        
        dataset_normalized.append(residual_absent)
        
        
        # rolling mean da sola 
        '''
        rolling_mean = pd.Series(column).rolling(int(16*seconds)).mean()
        rolling_mean.dropna(inplace=True)
        dataset_normalized.append(rolling_mean)'''
    
    train = np.array(dataset_normalized).transpose()
    
    ########## #################################### #############
    
    ##########EVALUATION OF ANOMALY SCORE##########
    
    model.fit(train)
    anomaly_scores = evaluate(model, np.concatenate((train, test)), w)
    ###############################################
    
    return anomaly_scores
    

anomaly_scores = feature_extraction(dataset_1)
plt.figure()
plt.plot(anomaly_scores)
plt.ylim(bottom = -0.05, top = 1.05)
plt.title("Anomaly Score on dataset")
plt.show()

anomaly_scores = feature_extraction(dataset)
plt.figure()
plt.plot(anomaly_scores)
plt.ylim(bottom = -0.05, top = 1.05)
plt.axvline(x=2221, c='red')
plt.axvline(x=2531, c='black')
plt.axvline(x=2531+1403, c='red')
plt.title("Anomaly Score on dataset")
plt.show()
