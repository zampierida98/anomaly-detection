# -*- coding: utf-8 -*-
"""
HMM voli aerei
 
"""

import pandas as pd
import numpy as np
from scipy import stats, linalg, integrate
from hmmlearn import hmm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

import warnings
warnings.filterwarnings('ignore')


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


def bic_fun(funz_likelihood, params, data):
    """
    Calcola la metrica per la scelta del modello all'interno di una classe di modelli
    
    -2 ln(L) + p*ln(len(data))
    
    - L è il valore max della funzione di likelihood ottenuta da model(HMM).score
    - p numero di parametri del modello HMM (Sezione: Structural architecture) https://en.wikipedia.org/wiki/Hidden_Markov_model 
    """
    
    bic = -2*funz_likelihood(data) + params*np.log(len(data))
    return bic


def chose_best_model(data, n_states_max=10):
    min_bic = float("inf")
    n_states = 0
    best_hmm = None
    
    print("Identintifico il miglior modello...")
    for N in range(1,n_states_max+1):

        print("Iterazione:", N, end=", ")
        
        hmm_candidate = hmm.GaussianHMM(n_components=N, covariance_type='diag', n_iter=100, random_state=0)
        hmm_candidate.fit(data)

        # Calcolo il numero parametri di HMM
        # - probabilità di transizione N*(N-1)
        # - distribuzione multivariata quindi MEDIE + MATRICE COVARIANZA = (N*M(M+3))/2
        #
        #(Sezione: Structural architecture) https://en.wikipedia.org/wiki/Hidden_Markov_model 
        
        # n_features: Dimensionality of the Gaussian emissions.
        M = hmm_candidate.n_features

        #parameters = N*(N-1) + (N*M*(M+3))/2
        parameters = M + M^2 + N*M + N*M

        bic = bic_fun(hmm_candidate.score, parameters, data)
        
        if bic < min_bic:
            min_bic = bic
            best_hmm = hmm_candidate
            n_states = N


        print("miglior modello:", n_states, ", valore bic=", min_bic)

    return (best_hmm, n_states)


################# MAIN
path_train = './datasets/dataset_voli/187093582_train_set.csv'
path_test  = './datasets/dataset_voli/187093582_test_set.csv'

train = pd.read_csv(path_train).values
test  = pd.read_csv(path_test).values

Y = test[:,3]
test = np.delete(test, np.s_[3], axis=1)

################# RIDUZIONE DEL RUMORE
# ho provato con rolling mean ma vengono risultati molto pessimi
# anche con decomposizione si ottengono dei risultati poco accettabili nonostante
# la migliore stagionalità sia 367
'''
train_normalized = []
for column in train.transpose():
    
    # decomposizione serie temporale
    ts_column = pd.Series(column)
    decomposition = seasonal_decompose(ts_column, model="additive", period=736) #367
    
    residual_absent = decomposition.seasonal + decomposition.trend
    residual_absent.dropna(inplace=True)        
    train_normalized.append(residual_absent)

train = np.array(train_normalized).transpose() # 1 - roll mean sul train set
'''

################# SCELTA MIGLIOR MODELLO
# fa già il fitting dei dati
#best_hmm, n_states = chose_best_model(train, 30)
#print("Il miglior modello è quello con", n_states)
#model = best_hmm

#K = 20
K = 30 # miglior modello secondo BIC
model = hmm.GaussianHMM(n_components=K, covariance_type="diag", n_iter=100, random_state=0)

model.fit(train)

# %% grandezza della finestra variabile
w_ott = 0
max_AUC = 0

for w in range(0,101,2):
    try:
        anomaly_scores = evaluate(model, test, w)
    except:
        continue
    
    fpr, tpr, t = roc_curve(Y[(w-1):], anomaly_scores)
    AUC = integrate.trapz(tpr, fpr)
    print("Finestra =", w, "AUC =", AUC)
    
    if AUC > max_AUC:
        w_ott = w
        max_AUC = AUC

# %% curva ROC e statistiche al variare della threshold
w = 96
#w = w_ott # finestra ottimale
anomaly_scores = evaluate(model, test, w)

plt.figure(dpi=125)
plt.plot(np.arange(0,240), anomaly_scores[0:240])
plt.fill_between(np.arange(0,240), Y[(w-1):240+(w-1)], color='red', alpha=0.5)
plt.ylim(bottom = -0.05, top = 1.05)
plt.title("Anomaly score (dataset voli) - {} stati e finestra = {}".format(K,w))
plt.xlabel("Observation")
plt.ylabel("Score")
plt.show()

# test e anomaly_scores hanno medesima lunghezza dato che non sono state fatte operazioni
# per ridurre il rumore
print(len(test))
print(len(anomaly_scores))

# label che evidenziano se c'è una ANOMALIA oppure NO
print(len(Y))

thresholds = [i for i in np.arange(0,1,0.05)]
precs, recs, f1s, accs, tprs, fprs = [], [], [], [], [], []

for thresh in thresholds:
    tp, fp, tn, fn = 0, 0, 0, 0
    
    for i in range(len(anomaly_scores)):
        s = anomaly_scores[i]
        if s > thresh: # valore anomalo?
            if Y[i+(w-1)] == True:
                tp += 1
            else:
                fp += 1
        if s < thresh: # valore nominale?
            if Y[i+(w-1)] == False: 
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
plt.figure(dpi=125)
plt.plot(thresholds, precs)
plt.title("Grafico di precision (dataset voli)")
plt.xlabel("Threshold")
plt.ylabel("Precision")
plt.show()

plt.figure(dpi=125)
plt.plot(thresholds, recs)
plt.title("Grafico di recall (dataset voli)")
plt.xlabel("Threshold")
plt.ylabel("Recall")
plt.show()

# https://en.wikipedia.org/wiki/F-score#Definition
plt.figure(dpi=125)
plt.plot(thresholds, f1s)
plt.title("Grafico di F1 score (dataset voli)")
plt.xlabel("Threshold")
plt.ylabel("F1 score")
plt.show()

# https://it.wikipedia.org/wiki/Receiver_operating_characteristic
plt.figure(dpi=125)
plt.plot(thresholds, accs)
plt.title("Grafico di accuracy (dataset voli)")
plt.xlabel("Threshold")
plt.ylabel("Accuracy")
plt.show()

plt.figure(dpi=125)
plt.plot(thresholds, tprs)
plt.title("Grafico del True Positive Rate (dataset voli)")
plt.xlabel("Threshold")
plt.ylabel("TPR")
plt.show()

plt.figure(dpi=125)
plt.plot(thresholds, fprs)
plt.title("Grafico del False Positive Rate (dataset voli)")
plt.xlabel("Threshold")
plt.ylabel("FPR")
plt.show()

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
fpr, tpr, t = roc_curve(Y[(w-1):], anomaly_scores)
plt.figure(dpi=125)
plt.plot(fpr, tpr)
plt.title("Curva ROC (dataset voli)")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()

AUC = integrate.trapz(tpr, fpr)

print("AUC (metodo integrazione trapezoidale):", AUC)
