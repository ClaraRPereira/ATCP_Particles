
# coding: utf-8

## Starting kit for the Higgs boson machine learning challenge

# This notebook contains a starting kit for the <a href="https://www.kaggle.com/c/higgs-boson">
# Higgs boson machine learning challenge</a>. Download the training set (called <code>training.csv</code>) and the test set (<code>test.csv</code>), then execute cells in order.

# In[1]:

import random,string,math,csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as GBC
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
#X_train, X_valid, Y_train, Y_valid = train_test_split( data_train[:,1:31], data_train[:,32], test_size = 0.1, random_state = 100)
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
gbc = GBC(n_estimators=100, max_depth=6,min_samples_leaf=200,max_features=30,verbose=1)
gbc.fit(X_train,Y_train) 

# #############################################################################
# Plot feature importance
feature_importance = gbc.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, nomes[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
# ############################################################################# 

# #############################################################################
# Plot training deviance

# compute validation set deviance
test_score = np.zeros((100,), dtype=np.float64)
for i, y_pred in enumerate(gbc.staged_predict(X_valid)):
    test_score[i] = gbc.loss_(Y_valid, y_pred)
plt.title('Deviance')
plt.plot(np.arange(100) + 1, gbc.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(100) + 1, test_score, 'r-',
         label='Validation Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
plt.show()
# #############################################################################


# #############################################################################
# Plot decision boundary

#X_plot = data_train[:,0][r<0.9]
#Y_plot = data_train[:,2][r<0.9]
#plot_step = 1
#x_min, x_max =  0, 400
#y_min, y_max =  0, 400
#x_min, x_max = X_plot.min() - 1, X_plot.max() + 1
#y_min, y_max = Y_plot.min() - 1, Y_plot.max() + 1

#xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))
#plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
#Z = gbc.predict_proba(X_valid)[:,1] 
###Z = gbc.predict(np.c_[xx.ravel(), yy.ravel()])
#Z = Z.reshape(xx.shape)
#cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
#plt.show()


#colors = ['b','r']

#plt.scatter(X_valid[:,0],X_valid[:,2], alpha=.1, s=4)
#plt.show()
# ############################################################################# 
 
# Get the probability output from the trained method, using the 10% for testing
prob_predict_train = gbc.predict_proba(X_train)[:,1]
prob_predict_valid = gbc.predict_proba(X_valid)[:,1]
 
# Experience shows me that choosing the top 15% as signal gives a good AMS score.
# This can be optimized though!
pcut = np.percentile(prob_predict_train,85)
 
# This are the final signal and background predictions
Yhat_train = prob_predict_train > pcut 
Yhat_valid = prob_predict_valid > pcut
 
# To calculate the AMS data, first get the true positives and true negatives
# Scale the weights according to the r cutoff.
TruePositive_train = W_train*(Y_train==1.0)*(1.0/0.9)
TrueNegative_train = W_train*(Y_train==0.0)*(1.0/0.9)
TruePositive_valid = W_valid*(Y_valid==1.0)*(1.0/0.1)
TrueNegative_valid = W_valid*(Y_valid==0.0)*(1.0/0.1)
 
# s and b for the training 
s_train = sum ( TruePositive_train*(Yhat_train==1.0) )
b_train = sum ( TrueNegative_train*(Yhat_train==1.0) )
s_valid = sum ( TruePositive_valid*(Yhat_valid==1.0) )
b_valid = sum ( TrueNegative_valid*(Yhat_valid==1.0) )
 
# Now calculate the AMS scores
print 'Calculating AMS score for a probability cutoff pcut=',pcut
def AMSScore(s,b): return  math.sqrt (2.*( (s + b + 10.)*math.log(1.+s/(b+10.))-s))
print '   - AMS based on 90% training   sample:',AMSScore(s_train,b_train)
print '   - AMS based on 10% validation sample:',AMSScore(s_valid,b_valid)
 
# Now we load the testing data, storing the data (X) and index (I)
print 'Loading testing data'
data_test = np.loadtxt( 'test.csv', delimiter=',', skiprows=1 )
X_test = data_test[:,1:31]
I_test = list(data_test[:,0])
 
# Get a vector of the probability predictions which will be used for the ranking
print 'Building predictions'
Predictions_test = gbc.predict_proba(X_test)[:,1]
# Assign labels based the best pcut
Label_test = list(Predictions_test>pcut)
Predictions_test =list(Predictions_test)
 
# Now we get the CSV data, using the probability prediction in place of the ranking
print 'Organizing the prediction results'
resultlist = []
for x in range(len(I_test)):
    resultlist.append([int(I_test[x]), Predictions_test[x], 's'*int((Label_test[x]==1.0))+'b'*int((Label_test[x]==0.0))])
 
# Sort the result list by the probability prediction
resultlist = sorted(resultlist, key=lambda a_entry: a_entry[1]) 
 
# Loop over result list and replace probability prediction with integer ranking
for y in range(len(resultlist)):
    resultlist[y][1]=y+1
 
# Re-sort the result list according to the index
resultlist = sorted(resultlist, key=lambda a_entry: a_entry[0])
 
# Write the result list data to a csv file
print 'Writing a final csv file Kaggle_higgs_prediction_output.csv'
fcsv = open('Kaggle_higgs_prediction_output.csv','w')
fcsv.write('EventId,RankOrder,Class\n')
for line in resultlist:
    theline = str(line[0])+','+str(line[1])+','+line[2]+'\n'
    fcsv.write(theline) 
fcsv.close()
