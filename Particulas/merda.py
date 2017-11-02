
# coding: utf-8

## Starting kit for the Higgs boson machine learning challenge

# This notebook contains a starting kit for the <a href="https://www.kaggle.com/c/higgs-boson">
# Higgs boson machine learning challenge</a>. Download the training set (called <code>training.csv</code>) and the test set (<code>test.csv</code>), then execute cells in order.

# In[1]:

import random,string,math,csv
import numpy as np
import matplotlib.pyplot as plt


### Reading an formatting training data

# In[2]:

all = list(csv.reader(open("training.csv","rb"), delimiter=','))


# Slicing off header row and id, weight, and label columns.

# In[3]:

xs = np.array([map(float, row[1:-2]) for row in all[1:]])
(numPoints,numFeatures) = xs.shape


# Perturbing features to avoid ties. It's far from optimal but makes life easier in this simple example.

# In[4]:

xs = np.add(xs, np.random.normal(0.0, 0.0001, xs.shape))


# Label selectors.

# In[5]:

sSelector = np.array([row[-1] == 's' for row in all[1:]])
bSelector = np.array([row[-1] == 'b' for row in all[1:]])


# Weights and weight sums.

# In[6]:

weights = np.array([float(row[-2]) for row in all[1:]])
sumWeights = np.sum(weights)
sumSWeights = np.sum(weights[sSelector])
sumBWeights = np.sum(weights[bSelector])

for i in range(numFeatures):
    plt.figure()
    Data = np.array([float(row[i]) for row in xs])
    sWeights = np.array(weights[sSelector])
    bWeights = np.array(weights[bSelector])
    bData = np.array(Data[bSelector])
    sData = np.array(Data[sSelector])


    # # Decorrelation test 1
# 
# Description: performs standardization + PCA on the input data as a whole and runs it through the MVA
# 
# Results: Performance is tremendously increased, reaching very close to saturation (ROC AUC ~ 0.9999) for the worst case scenario (fully hadronic decay).
# 
# Method is applicable to real data (see Decorrelation tests 4 & 5)
# 

### Import modules


from rep.estimators import XGBoostClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import root_pandas
from root_pandas import read_root
import math
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score
import time

### Import data

signalData = bData
backgroundData = sData
print "Samples contains", len(signalData), "signal events and", len(backgroundData), "background events"
print len(signalData)+len(backgroundData), "events in total"


### Add classification targets and combine into single dataset

signalData["target"] = 1
backgroundData["target"] = 0
data = signalData.append(backgroundData, ignore_index = True)

gencols = [gen for gen in signalData.columns if str.startswith(gen, "gen")]
data.drop(gencols,axis = 1,inplace=True)
# removes the generator level variables, used to train a regression algorithm


### Standardize input data

allFeatures = data.columns-["target"]
#ignores the target variable 

data_std = StandardScaler().fit_transform(data.ix[:,0:len(allFeatures)].values)
#standardizes input data for all variables


### Perform principal component analysis

ncomp = len(allFeatures)
sklearn_pca = sklearnPCA(n_components = ncomp)
# initiates an instance of a PCA -> dimension of sub-space is the same as the original space

pcaaux = sklearn_pca.fit_transform(data_std)
"""
Fits the PCA (finds the n orthogonal vectors that represent the directions of maximum variance in the data) 
and projects the original dataset in the new axes (transform)
"""

cols = ["var" + str(i) for i in range(0,ncomp)]
pcadf = pd.DataFrame(np.float64(pcaaux),index=data.index,columns=cols)
pcadf = pd.concat([pcadf, data['target']], axis=1)
# transforms the NumPy array output from PCA into a Pandas dataframe (ease of use)


### Define classifier

xgbc = XGBoostClassifier()


### Cross validation

# Runs many training loops and outputs mean score and uncertainty; shows whether your sweet new high-level variable actually makes a difference

start = time.time()
varnames = [var for var in pcadf.columns if var != 'target']
print varnames
xgbcCV = cross_val_score(xgbc, pcadf[varnames].astype(np.float64), 
                         pcadf["target"].astype(np.bool), cv=4, scoring="roc_auc")
print "ROC AUC: {:.4f} (+/- {:.4f})".format(xgbcCV.mean(), xgbcCV.std()/math.sqrt(len(xgbcCV)))
print "Cross-validation took {:.3f}s ".format(time.time() - start)


### Randomly split data into training and validation samples

trainData, valData = train_test_split(pcadf, random_state=11, train_size=0.5)


### Train classifier

start = time.time()
xgbc.fit(trainData[varnames].astype(np.float64), trainData.target.astype(np.bool))
print "Fitting took {:.3f}s ".format(time.time() - start) 


### Test response on validation data and print ROC AUC

probVal = xgbc.predict_proba(valData[varnames].astype(np.float64))
area = roc_auc_score(valData.target, probVal[:, 1])
print "ROC AUC", area
plt.figure(figsize=[8, 8])
plt.plot(*roc_curve(valData.target, probVal[:, 1])[:2], label='Validation')
plt.plot([0, 1], [0, 1], 'k--', label='No discrimination')
plt.xlabel('Background acceptance'), plt.ylabel('Signal acceptance')
plt.legend(loc='best')
plt.show()