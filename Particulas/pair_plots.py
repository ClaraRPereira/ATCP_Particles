import random,string,math,csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time

sns.set(style="ticks", color_codes=True)


### Reading an formatting training data
all = list(csv.reader(open("training.csv","rb"), delimiter=','))


# Slicing off header row and id, weight, and label columns.
xs = np.array([map(float, row[1:-2]) for row in all[1:]])
(numPoints,numFeatures) = xs.shape

# Label selectors.
sSelector = np.array([row[-1] == 's' for row in all[1:]])
bSelector = np.array([row[-1] == 'b' for row in all[1:]])

for i in range(numFeatures):
	for j in range(numFeatures):

		plt.figure()

		Data = np.array([float(row[i]) for row in xs])
		bData = np.array(Data[bSelector])
		sData = np.array(Data[sSelector])

		Data2 = np.array([float(row[j]) for row in xs])
		bData2 = np.array(Data2[bSelector])
		sData2 = np.array(Data2[sSelector])

		indexb = (bData >= -900) & (bData2 >= -900) 
		indexs = (sData >= -900) & (sData2 >= -900)  

		s = 2
		plt.scatter(bData[indexb], bData2[indexb], color='r', s=s, alpha=.4, label="Noise")
		plt.scatter(sData[indexs], sData2[indexs], color='b', s=s, alpha=.4, label="H")
		plt.xlabel(str(all[0][i+1]))
		plt.ylabel(str(all[0][j+1]))
		plt.legend(prop={'size':9})
		plt.savefig(str(all[0][i+1])+"_vs_"+str(all[0][j+1]))
		plt.show()

