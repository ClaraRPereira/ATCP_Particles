
# coding: utf-8

## Starting kit for the Higgs boson machine learning challenge

# This notebook contains a starting kit for the <a href="https://www.kaggle.com/c/higgs-boson">
# Higgs boson machine learning challenge</a>. Download the training set (called <code>training.csv</code>) and the test set (<code>test.csv</code>), then execute cells in order.

# In[1]:

import random,string,math,csv
import numpy as np
import matplotlib.pyplot as plt


def check_submission(submission, Nelements):
    """ Check that submission RankOrder column is correct:
        1. All numbers are in [1,NTestSet]
        2. All numbers are unqiue
    """
    rankOrderSet = set()    
    with open(submission, 'rb') as f:
        sub = csv.reader(f)
        sub.next() # header
        for row in sub:
            rankOrderSet.add(row[1])
            
    if len(rankOrderSet) != Nelements:
        print 'RankOrder column must contain unique values'
        exit()
    elif rankOrderSet.isdisjoint(set(xrange(1,Nelements+1))) == False:
        print 'RankOrder column must contain all numbers from [1..NTestSset]'
        exit()
    else:
        return True

def AMS(s, b):
    """ Approximate Median Significance defined as:
        AMS = sqrt(
                2 { (s + b + b_r) log[1 + (s/(b+b_r))] - s}
              )        
    where b_r = 10, b = background, s = signal, log is natural logarithm """
    
    br = 10.0
    radicand = 2 *( (s+b+br) * math.log (1.0 + s/(b+br)) -s)
    if radicand < 0:
        print 'radicand is negative. Exiting'
        exit()
    else:
        return math.sqrt(radicand)

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

### Training and validation cuts

# We will train a classifier on a random training set for minimizing the weighted error with balanced weights, then we will maximize the AMS on the held out validation set.

# In[7]:

randomPermutation = random.sample(range(len(xs)), len(xs))
numPointsTrain = int(numPoints*0.9)
numPointsValidation = numPoints - numPointsTrain

xsTrain = xs[randomPermutation[:numPointsTrain]]
xsValidation = xs[randomPermutation[numPointsTrain:]]

sSelectorTrain = sSelector[randomPermutation[:numPointsTrain]]
bSelectorTrain = bSelector[randomPermutation[:numPointsTrain]]
sSelectorValidation = sSelector[randomPermutation[numPointsTrain:]]
bSelectorValidation = bSelector[randomPermutation[numPointsTrain:]]

weightsTrain = weights[randomPermutation[:numPointsTrain]]
weightsValidation = weights[randomPermutation[numPointsTrain:]]

sumWeightsTrain = np.sum(weightsTrain)
sumSWeightsTrain = np.sum(weightsTrain[sSelectorTrain])
sumBWeightsTrain = np.sum(weightsTrain[bSelectorTrain])


# In[8]:

xsTrainTranspose = xsTrain.transpose()


# Making signal and background weights sum to $1/2$ each to emulate uniform priors $p(s)=p(b)=1/2$.

# In[9]:

weightsBalancedTrain = np.array([0.5 * weightsTrain[i]/sumSWeightsTrain
                                 if sSelectorTrain[i]
                                 else 0.5 * weightsTrain[i]/sumBWeightsTrain\
                                 for i in range(numPointsTrain)])


### Training naive Bayes and defining the score function

# Number of bins per dimension for binned naive Bayes.

# In[10]:

numBins = 10


# <code>logPs[fI,bI]</code> will be the log probability of a data point <code>x</code> with <code>binMaxs[bI - 1] < x[fI] <= binMaxs[bI]</code> (with <code>binMaxs[-1] = -</code>$\infty$ by convention) being a signal under uniform priors $p(\text{s}) = p(\text{b}) = 1/2$.

# In[11]:

logPs = np.empty([numFeatures, numBins])
binMaxs = np.empty([numFeatures, numBins])
binIndexes = np.array(range(0, numPointsTrain+1, numPointsTrain/numBins))


# In[12]:

for fI in range(numFeatures):
    # index permutation of sorted feature column
    indexes = xsTrainTranspose[fI].argsort()

    for bI in range(numBins):
        # upper bin limits
        binMaxs[fI, bI] = xsTrainTranspose[fI, indexes[binIndexes[bI+1]-1]]
        # training indices of points in a bin
        indexesInBin = indexes[binIndexes[bI]:binIndexes[bI+1]]
        # sum of signal weights in bin
        wS = np.sum(weightsBalancedTrain[indexesInBin]
                    [sSelectorTrain[indexesInBin]])
        # sum of background weights in bin
        wB = np.sum(weightsBalancedTrain[indexesInBin]
                    [bSelectorTrain[indexesInBin]])
        # log probability of being a signal in the bin
        logPs[fI, bI] = math.log(wS/(wS+wB))


# The score function we will use to sort the test examples. For readability it is shifted so negative means likely background (under uniform prior) and positive means likely signal. <code>x</code> is an input vector.

# In[13]:

def score(x):
    logP = 0
    for fI in range(numFeatures):
        bI = 0
        # linear search for the bin index of the fIth feature
        # of the signal
        while bI < len(binMaxs[fI]) - 1 and x[fI] > binMaxs[fI, bI]:
            bI += 1
        logP += logPs[fI, bI] - math.log(0.5)
    return logP


# In[42]:

test = list(csv.reader(open("test.csv", "rb"),delimiter=','))
xsTest = np.array([map(float, row[1:]) for row in test[1:]])


# In[43]:
DER_M_MMC =np.array([float(row[0]) for row in xsTest])
DER_MT_MET_LEP =np.array([float(row[1]) for row in xsTest])
DER_M_vis =np.array([float(row[2]) for row in xsTest])
DER_deltaeta_jj =np.array([float(row[4]) for row in xsTest])
testIds = np.array([int(row[0]) for row in test[1:]])

#print testIds
print len(testIds)

# Computing the scores.

# In[44]:

#aux= random.sample(1,550000)
#testScores= np.array([random.sample(xrange(1,550000), len(xsTest))])
#testScores = np.array([random.randint(1,550000) for x in xsTest])


#random.shuffle(testScores)

#data = range(1, 550001)
#print len(data)
#random.shuffle(data)

	

testScores = np.zeros(len(DER_M_MMC))
for i in range(1,len(DER_M_MMC)):
	if DER_M_MMC[i]>=95 and DER_M_MMC[i]<=165 and DER_MT_MET_LEP[i] >=0 and DER_MT_MET_LEP[i]<=60 and DER_M_vis[i]>35 and DER_M_vis[i]<120:
		testScores[i]=1
	else:
		testScores[i]=0

#data = range(1, 550001)
#print len(data)
#random.shuffle(data)

data = range(1, 550001)
print len(data)
random.shuffle(data)

testScores2 = np.zeros(len(data))
for i in range(len(data)):
	testScores2[i]=int(data[i])

# Computing the rank order.

# In[45]:

#testScores2 = np.array([score(x) for x in xsTest])
testInversePermutation = testScores2.argsort()


# In[46]:

testPermutation = list(testInversePermutation)
for tI,tII in zip(range(len(testInversePermutation)),
                  testInversePermutation):
    testPermutation[tII] = tI


threshold=0
#threshold2=165

# Computing the submission file with columns EventId, RankOrder, and Class.
#str(testPermutation[tI]+1)
#
# In[47]:

submission = np.array([[str(testIds[tI]),str(testPermutation[tI]+1),
                       's' if testScores[tI] > threshold else 'b'] 
            for tI in range(len(testIds))])


# In[48]:

submission = np.append([['EventId','RankOrder','Class']],
                        submission, axis=0)



# In[49]:

np.savetxt("submissionCut.csv",submission,fmt='%s',delimiter=',')


check_submission("submissionCut.csv",550000)
# Saving the file that can be submitted to Kaggle.


