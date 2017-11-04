
import random,string,math,csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn.cluster as cluster
import time

sns.set(style="ticks", color_codes=True)


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

print (numFeatures)
np.random.seed(19680801)
# Plot Weights
#for i in range(numFeatures
i=1
plt.figure()
Data = np.array([float(row[i]) for row in xs])
sWeights = np.array(weights[sSelector])
bWeights = np.array(weights[bSelector])
bData = np.array(Data[bSelector])
sData = np.array(Data[sSelector])
#bData = bData[(bData >= -900)]
#sData = sData[(sData >= -900)]

j=13
Data2 = np.array([float(row[j]) for row in xs])
sWeights2 = np.array(weights[sSelector])
bWeights2 = np.array(weights[bSelector])
bData2 = np.array(Data2[bSelector])
sData2 = np.array(Data2[sSelector])
#bData2 = bData2[(bData2 >= -900)]
#sData2 = sData2[(sData2 >= -900)]
#plt.hist(bData,bins = "sqrt", normed = True, histtype = "step", label="Noise")
#plt.hist(sData,bins = "sqrt", normed = True, histtype = "step", label="H")
#plt.xlim(0,10)
plt.title(str(all[0][i+1]+all[0][j+1]) )

#plt.savefig(str(all[0][i+1]))
#plt.show()
#sns.set_context('poster')
#sns.set_color_codes()
#plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':1}


s = 2

plt.scatter(bData, bData2, color='r', s=s, alpha=.4, label="Noise")
plt.scatter(sData, sData2, color='b', s=s, alpha=.4, label="H")
plt.legend(prop={'size': 9})
plt.xlim(0, 300)
plt.ylim(0, 300)
plt.savefig(str(all[0][i+1]+all[0][j+1]))

#plt.scatter(xs.T[0], xs.T[1], c=colours, **plot_kwds)

#tips= np.array([map(float, row[1:3]) for row in all[1:]])
#iris = pd.read_csv('training.csv')
#tips=float(tips)
#sns.set(style="ticks")
#np.random.seed(sum(map(ord, "axis_grids")))


#g = sns.FacetGrid(int(tips), col="sex", hue="smoker")
#g.map(plt.scatter, "total_bill", "tip", alpha=.7)
#g.add_legend();
#iris = sns.load_dataset(tips2)
#g = sns.pairplot(iris, hue="species")
plt.show()