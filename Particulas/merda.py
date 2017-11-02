
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


# Plot Weights
#for i in range(numFeatures
i=0
plt.figure()
Data = np.array([float(row[i]) for row in xs])
sWeights = np.array(weights[sSelector])
bWeights = np.array(weights[bSelector])
bData = np.array(Data[bSelector])
sData = np.array(Data[sSelector])
bData = bData[(bData >= -900)]
sData = sData[(sData >= -900)]
#plt.hist(bData,bins = "sqrt", normed = True, histtype = "step", label="Noise")
#plt.hist(sData,bins = "sqrt", normed = True, histtype = "step", label="H")
#plt.xlim(0,300)
#plt.title(str(all[0][i+1]))
#plt.legend()
#plt.savefig(str(all[0][i+1]))
#plt.show()




# In[42]:

test = list(csv.reader(open("test.csv", "rb"),delimiter=','))
xsTest = np.array([map(float, row[1:]) for row in test[1:]])


# In[43]:
#DataTest = np.array([float(row[i]) for row in xsTest])
testIds = np.array([int(row[0]) for row in test[1:]])




# Computing the scores.

# In[44]:

#aux= random.sample(1,550000)
#testScores= np.array([random.sample(xrange(1,550000), len(xsTest))])
#testScores = np.array([random.randint(1,550000) for x in xsTest])


#random.shuffle(testScores)

#data = range(1, 550001)
#print len(data)
#random.shuffle(data)

	

testScores = np.zeros(len(test))
for i in range(len(test)):
	if test[i]>=900:
		testScores[testIds[i]]=0
	else:
		testScores[testIds[i]]=1


#a = range(1,500000)
#random.shuffle(a)
#testScores=np.array([aux for x in xsTest])

# Computing the rank order.

# In[45]:

testInversePermutation = testScores.argsort()


# In[46]:

testPermutation = list(testInversePermutation)
for tI,tII in zip(range(len(testInversePermutation)),
                  testInversePermutation):
    testPermutation[tII] = tI


threshold1=95
threshold2=165

# Computing the submission file with columns EventId, RankOrder, and Class.
#str(testPermutation[tI]+1)
#
# In[47]:

submission = np.array([[str(testIds[tI]),str(data[tI]),
                       's' if testScores[tI] >= threshold1 else 'b'] 
            for tI in range(len(testIds))])


# In[48]:

submission = np.append([['EventId','RankOrder','Class']],
                        submission, axis=0)


# Saving the file that can be submitted to Kaggle.

# In[49]:

np.savetxt("submission2.csv",submission,fmt='%s',delimiter=',')