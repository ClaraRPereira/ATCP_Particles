import random,string,math,csv
import numpy as np
import matplotlib.pyplot as plt

from random import randrange


### Reading an formatting training data
all = list(csv.reader(open("training.csv","rb"), delimiter=','))


# Slicing off header row and id, weight, and label columns.
xs = np.array([map(float, row[1:-2]) for row in all[1:]])
(numPoints,numFeatures) = xs.shape

xs2 = np.array([map(float, row[0:-2]) for row in all[1:]])



# Perturbing features to avoid ties. It's far from optimal but makes life easier in this simple example.
xs = np.add(xs, np.random.normal(0.0, 0.0001, xs.shape))


# Label selectors.
sSelector = np.array([row[-1] == 's' for row in all[1:]])
bSelector = np.array([row[-1] == 'b' for row in all[1:]])

# Weights and weight sums.
weights = np.array([float(row[-2]) for row in all[1:]])
sumWeights = np.sum(weights)
sumSWeights = np.sum(weights[sSelector])
sumBWeights = np.sum(weights[bSelector])

print numFeatures
# Plot Weights
for i in range(numFeatures):
	Data = np.array([float(row[i]) for row in xs])
	sWeights = np.array(weights[sSelector])
	bWeights = np.array(weights[bSelector])
	bData = np.array(Data[bSelector])
	sData = np.array(Data[sSelector])




	


# In[42]:

test = list(csv.reader(open("test.csv", "rb"),delimiter=','))
xsTest = np.array([map(float, row[1:]) for row in test[1:]])


# In[43]:

testIds = np.array([int(row[0]) for row in test[1:]])




# Computing the scores.

# In[44]:

#aux= random.sample(1,550000)
#testScores= np.array([random.sample(xrange(1,550000), len(xsTest))])
#testScores = np.array([random.randint(1,550000) for x in xsTest])


#random.shuffle(testScores)

data = range(1, 550001)
print len(data)
random.shuffle(data)

testScores = np.zeros(len(data))
for i in range(len(data)):
	testScores[i]=int(data[i])


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


threshold=275000

# Computing the submission file with columns EventId, RankOrder, and Class.
#str(testPermutation[tI]+1)
#
# In[47]:

submission = np.array([[str(testIds[tI]),str(data[tI]),
                       's' if testScores[tI] >= threshold else 'b'] 
            for tI in range(len(testIds))])


# In[48]:

submission = np.append([['EventId','RankOrder','Class']],
                        submission, axis=0)


# Saving the file that can be submitted to Kaggle.

# In[49]:

np.savetxt("submission2.csv",submission,fmt='%s',delimiter=',')