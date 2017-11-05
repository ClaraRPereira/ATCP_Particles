
# coding: utf-8

## Starting kit for the Higgs boson machine learning challenge

# This notebook contains a starting kit for the <a href="https://www.kaggle.com/c/higgs-boson">
# Higgs boson machine learning challenge</a>. Download the training set (called <code>training.csv</code>) and the test set (<code>test.csv</code>), then execute cells in order.

# In[1]:

import random,string,math,csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.metrics import roc_curve
from sklearn.cross_validation import train_test_split
import math
 
 
# Load training data
print 'Loading training data.'
data_train = np.loadtxt( 'training.csv', delimiter=',', skiprows=1, converters={32: lambda x:int(x=='s'.encode('utf-8')) } )

nomes = ['DER_mass_MMC','DER_mass transverse_met_lep','DER_mass_vis','DER_pt_h','DER_deltaeta_jet_jet','DER_mass_jet_jet','DER_prodeta_jet_jet','DER_deltar_tau_lep','DER_pt_tot','DER_sum_pt','DER_pt_ratio_lep_tau','DER_met_phi_centrality','DER_lep_eta_centrality','PRI_tau_pt','PRI_tau_eta','PRI_tau_phi','PRI_lep_pt','PRI_lep_eta','PRI_lep_phi','PRI_met','PRI_met_phi','PRI_met_sumet','PRI_jet_num','PRI_jet_leading_pt','PRI_jet_leading_eta','PRI_jet_leading_phi','PRI_jet_subleading_pt','PRI_jet_subleading_eta','PRI_jet_subleading_phi','PRI_jet_all_pt'] 
nomes = np.array(nomes)
for i in range(0,len(nomes)): 
	print str(nomes[i])
 
 
# Pick a random seed for reproducible results. Choose wisely!
np.random.seed(42)
# Random number for training/validation splitting
r =np.random.rand(data_train.shape[0])
 
# Put Y(truth), X(data), W(weight), and I(index) into their own arrays
print 'Assigning data to numpy arrays.'

# Split our data into training and test set.
# First 90% are training 
Y_train = data_train[:,32][r<0.9]
X_train = data_train[:,1:31][r<0.9]
W_train = data_train[:,31][r<0.9]
# First 10% are validation
Y_valid = data_train[:,32][r>=0.9]  # Extracting s/b value
X_valid = data_train[:,1:31][r>=0.9] # Extracting all feature values
W_valid = data_train[:,31][r>=0.9] # Extracting weights in training set



 
# Train the GradientBoostingClassifier using our good features
print 'Training classifier (this may take some time!)'
print 'learning 0.2'
gbc02 = GBC(learning_rate=0.2,n_estimators=50, max_depth=6,min_samples_leaf=200,max_features=30,verbose=1)
gbc02.fit(X_train,Y_train) 

print 'learning 0.1'
gbc01 = GBC(learning_rate=0.1,n_estimators=50, max_depth=6,min_samples_leaf=200,max_features=30,verbose=1)
gbc01.fit(X_train,Y_train) 

print 'learning 0.05'
gbc005 = GBC(learning_rate=0.05,n_estimators=50, max_depth=6,min_samples_leaf=200,max_features=30,verbose=1)
gbc005.fit(X_train,Y_train) 

print 'learning 0.02'
gbc002 = GBC(learning_rate=0.02,n_estimators=50, max_depth=6,min_samples_leaf=200,max_features=30,verbose=1)
gbc002.fit(X_train,Y_train) 


print 'subsamples 0.5'
gbcsub = GBC(learning_rate=0.1,n_estimators=50, subsample=0.5, max_depth=6,min_samples_leaf=200,max_features=30,verbose=1)
gbcsub.fit(X_train,Y_train) 


# #############################################################################
# ROC
# The gradient boosted model by itself
y_pred_gbc02 = gbc02.predict_proba(X_valid)[:, 1]
fpr_gbc02, tpr_gbc02, _ = roc_curve(Y_valid, y_pred_gbc02)


y_pred_gbc01 = gbc01.predict_proba(X_valid)[:, 1]
fpr_gbc01, tpr_gbc01, _ = roc_curve(Y_valid, y_pred_gbc01)

y_pred_gbc005 = gbc005.predict_proba(X_valid)[:, 1]
fpr_gbc005, tpr_gbc005, _ = roc_curve(Y_valid, y_pred_gbc005)


y_pred_gbc002 = gbc002.predict_proba(X_valid)[:, 1]
fpr_gbc002, tpr_gbc002, _ = roc_curve(Y_valid, y_pred_gbc002)


y_pred_gbcsub = gbcsub.predict_proba(X_valid)[:, 1]
fpr_gbcsub, tpr_gbcsub, _ = roc_curve(Y_valid, y_pred_gbcsub)


plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_gbc02, tpr_gbc02,label='learning 0.2')
plt.plot(fpr_gbc01, tpr_gbc01, label='learning 0.1')
plt.plot(fpr_gbcsub, tpr_gbcsub, label='learning 0.1 subsampling')
plt.plot(fpr_gbc005, tpr_gbc005, label='learning 0.05')
plt.plot(fpr_gbc002, tpr_gbc002, label='learning 0.02')

plt.legend(loc='lower right')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.savefig("ROC_curve_var_learning_rate.png",format='png')
plt.show()

# #############################################################################


# #############################################################################
# Plot training deviance
# compute validation set deviance

#test_score = np.zeros((50,), dtype=np.float64)
#for i, y_pred in enumerate(gbc.staged_decision_function(X_valid)):
#    test_score[i] = gbc.loss_(Y_valid, y_pred)
    
plt.title('Deviance')
plt.plot(np.arange(50) + 1, gbc02.train_score_,label='learning 0.2')
plt.plot(np.arange(50) + 1, gbc01.train_score_,label='learning 0.1')
plt.plot(np.arange(50) + 1, gbcsub.train_score_,label='learning 0.1 subsampling')
plt.plot(np.arange(50) + 1, gbc005.train_score_,label='learning 0.05')
plt.plot(np.arange(50) + 1, gbc002.train_score_,label='learning 0.02')
#plt.plot(np.arange(50) + 1, test_score, 'r-',label='Validation Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Training Set Deviance')
plt.savefig("Deviance_var_learnign_rate.png",format='png')
plt.show()

