
import random,string,math,csv
import numpy as np
import matplotlib.pyplot as plt


### Reading an formatting training data
all = list(csv.reader(open("training.csv","rb"), delimiter=','))


# Slicing off header row and id, weight, and label columns.
xs = np.array([map(float, row[1:-2]) for row in all[1:]])
(numPoints,numFeatures) = xs.shape


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
#for i in range(numFeatures
i=4
plt.figure()
Data = np.array([float(row[i]) for row in xs])
sWeights = np.array(weights[sSelector])
bWeights = np.array(weights[bSelector])
bData = np.array(Data[bSelector])
sData = np.array(Data[sSelector])
bData = bData[(bData >= -900)]
sData = sData[(sData >= -900)]
plt.hist(bData,bins = "sqrt", normed = True, histtype = "step", label="Noise")
plt.hist(sData,bins = "sqrt", normed = True, histtype = "step", label="H")
plt.xlim(0,10)
plt.title(str(all[0][i+1]))
plt.legend()
#plt.savefig(str(all[0][i+1]))
plt.show()

