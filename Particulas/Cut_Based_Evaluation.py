
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


Y_pred = np.zeros(len(Y_valid))
TP = np.zeros(len(Y_valid))
FN = np.zeros(len(Y_valid))
FP = np.zeros(len(Y_valid))
TN = np.zeros(len(Y_valid))



for i in range(0,len(Y_valid)):
	if X_valid[i,0] >= 95 and X_valid[i,0] <= 165 and X_valid[i,1] >= 0 and X_valid[i,1] <= 60:
		Y_pred[i]=1
	else:	
		Y_pred[i]=0	
#print 'Total dados\n' 
#print Y_valid.size
#print 'Total sinal\n' 
#print np.sum(Y_valid)
#print '\nTotal sinal previsto\n' 
#print np.sum(Y_pred)



for i in range(0,len(Y_valid)):
	if Y_valid[i] == 1 and  Y_pred[i] == 1 :
		TP[i]=1	
	elif Y_valid[i] == 1 and  Y_pred[i] == 0 :
		FN[i]=1
	elif Y_valid[i] == 0 and  Y_pred[i] == 1 :	
		FP[i]=1		
	else : 
		TN[i]=1	
		
tp=np.sum(TP) 
fp=np.sum(FP)
tn=np.sum(TN)
fn=np.sum(FN)
		
sensitivity = tp/(tp+fn)
specificity = tn/(fp+tn)		
precision = tp/(tp+fp)
accuracy = (tp+tn)/(tp+fn+fp+tn)
F1 = 2*(precision*sensitivity)/(precision+sensitivity)

print '\n\nTrue Positive {} \tFalse Positive {}'.format(tp,fp)		
print 'False Negative {} \tTrue Negative {}'.format(fn,tn)				

print '\nSensitivity TPR:\t{:.4f}'.format(sensitivity)
print 'Specificity TNR:\t{:.4f}'.format(specificity)
print 'Precision:\t\t{:.4f}'.format(precision)
print 'Accuracy:\t\t{:.4f}'.format(accuracy)	
print 'F1-Measure:\t\t{:.4f}'.format(F1)	

