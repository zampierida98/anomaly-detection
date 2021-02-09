# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy import stats, linalg
from hmmlearn import hmm
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

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
w = 100
K = 5
###############################################

##########DATA AND MODEL TRAINING##############
dataset = pd.read_csv(path_data).values
###############################################


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

# %% 1 - roll mean sul train set
# la testa del dataset che non veniva valutata per la decomposizione
head = dataset[0:107,feat]

train = dataset[0:2221,feat]

#la decomposizione con season a 200 toglie i 100 campioni all'inizio e alla fine
test = dataset[2121:,feat]
'''
# %% 2a - train set con 85% delle osservazioni
train = dataset[0:2151,feat]
test = dataset[2051:,feat]

# %% 2b - train set con 90% delle osservazioni
train = dataset[0:2278,feat]
test = dataset[2178:,feat]

# %% 3 - roll mean sull'intero dataset
dataset = dataset[:,feat]
'''
#%%######### RIDUZIONE DEL RUMORE CON ROLLING MEAN ############
seconds = 0.5

dataset_normalized = []
for column in train.transpose(): # 1 - roll mean sul train set
#for column in dataset.transpose(): # 3 - roll mean sull'intero dataset
    
    # rolling mean 
    column = pd.Series(column).rolling(int(16*seconds)).mean()
    column.dropna(inplace=True)
    
    # decomposizione serie temporale
    ts_column = pd.Series(column)
    decomposition = seasonal_decompose(ts_column, model="additive", period=200)
    
    residual_absent = decomposition.seasonal + decomposition.trend
    residual_absent.dropna(inplace=True)        
    dataset_normalized.append(residual_absent)

train = np.array(dataset_normalized).transpose() # 1 - roll mean sul train set
#dataset = np.array(dataset_normalized).transpose() # 3 - roll mean sull'intero dataset
#train = dataset[0:2121] # 3 - roll mean sull'intero dataset
############ #################################### #############

##########EVALUATION OF ANOMALY SCORE##########
model.fit(train)

anomaly_scores = evaluate(model, np.concatenate((train, test)), w)
#anomaly_scores = evaluate(model, np.concatenate((head, train, test)), w)
#anomaly_scores = evaluate(model, test, w)

#anomaly_scores = evaluate(model, dataset, w) # 3 - roll mean sull'intero dataset

plt.figure()
plt.plot(anomaly_scores)
plt.ylim(bottom = -0.05, top = 1.05)
plt.title("Anomaly Score on dataset")
plt.show()
###############################################

# %% curva ROC e statistiche al variare della threshold
print(len(dataset)) # 2531
print(len(anomaly_scores)) # 2325 + 106 persi in rolling mean + 100 persi in evaluate = 2531
print(len(Y)) # Y usato da 206 a 2531

thresholds = [i for i in np.arange(0.3,0.82,0.02)] # 26 valori da 0.3 a 0.8
precs, recs, f1s, accs, tprs, fprs = [], [], [], [], [], []

for thresh in thresholds:
    tp, fp, tn, fn = 0, 0, 0, 0
    
    for i in range(len(anomaly_scores)):
        s = anomaly_scores[i]
        if s > thresh: # valore anomalo?
            if Y[i+206] == True:
                tp += 1
            else:
                fp += 1
        if s < thresh: # valore nominale?
            if Y[i+206] == False:
                tn += 1
            else:
                fn += 1
    '''
    print('Precision:', tp/(tp+fp))
    print('Recall:', tp/(tp+fn))
    print('F1 score', tp / (tp + 1/2 * (fp+fn)))
    print('Accuracy', (tp+tn) / ((tp+fn)+(fp+tn)))
    print('TPR', tp / (tp+fn))
    print('FPR', fp / (fp+tn))
    '''
    prec = tp/(tp+fp)
    rec = tp/(tp+fn)
    f1 = tp / (tp + 1/2 * (fp+fn))
    acc = (tp+tn) / ((tp+fn)+(fp+tn))
    tpr = tp / (tp+fn)
    fpr = fp / (fp+tn)
    
    precs.append(prec)
    recs.append(rec)
    f1s.append(f1)
    accs.append(acc)
    tprs.append(tpr)
    fprs.append(fpr)

# https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)
plt.figure()
plt.plot(thresholds, precs)
plt.title("Precision")
plt.show()

plt.figure()
plt.plot(thresholds, recs)
plt.title("Recall")
plt.show()

# https://en.wikipedia.org/wiki/F-score#Definition
plt.figure()
plt.plot(thresholds, f1s)
plt.title("F1 score")
plt.show()

# https://it.wikipedia.org/wiki/Receiver_operating_characteristic
plt.figure()
plt.plot(thresholds, accs)
plt.title("Accuracy")
plt.show()

plt.figure()
plt.plot(thresholds, tprs)
plt.title("True Positive Rate")
plt.show()

plt.figure()
plt.plot(thresholds, fprs)
plt.title("False Positive Rate")
plt.show()

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
fpr, tpr, t = roc_curve(Y[206:], anomaly_scores)
plt.figure()
plt.plot(fpr, tpr)
plt.title("Curva ROC")
plt.show()
