

import numpy as np
import os
import functools
from operator import mul

basePath = "data/custom/labels/"
k=12 #Choose cluster size
clusterList = []

#Read in a list of bounding box positions for data set
for animal in os.listdir(basePath):
    for file in os.listdir(basePath+"/"+animal):

        info = np.loadtxt(basePath+"/"+animal+"/"+file)     
        if np.size(info)==5: 
            xDiff = np.abs(info[3]-info[1])
            yDiff = np.abs(info[4] - info[2])
            if xDiff != 0 and yDiff != 0: #A few images seem to be mislabeled, resulting in faulty box coordinates. We remove them here.
                clusterList.append([xDiff, yDiff])
        else:
            for row in range(0,np.size(info,0)):
                 xDiff = np.abs(info[row,3]-info[row,1])
                 yDiff = np.abs(info[row,4] - info[row,2])
                 if xDiff != 0 and yDiff != 0:
                     clusterList.append([xDiff, yDiff])

anchorBoxList = np.array(clusterList)
nrOfBoxes = anchorBoxList.shape[0]
dist = np.empty((nrOfBoxes, k))

outputCurrent = np.zeros((nrOfBoxes,))
outputSmallestError = np.ones((nrOfBoxes,))
output = anchorBoxList[np.random.choice(nrOfBoxes, k, replace=False)]

# Run k-means with IoU as distance function until convergence
while np.sum(outputSmallestError-outputCurrent != 0):
    outputCurrent = outputSmallestError
    for row in range(nrOfBoxes):
       
        #Calculate IoU
        i = np.minimum(output[:, 0], anchorBoxList[row,0])*np.minimum(output[:, 1], anchorBoxList[row,1])         
        a_box = anchorBoxList[row,0]*anchorBoxList[row,1]
        a_cluster = output[:, 0] * output[:, 1]
    
        # Use IoU as distance function, as described in article
        iou = i / (a_box + a_cluster - i)
        
        dist[row] = 1 - iou

    outputSmallestError = np.argmin(dist, axis=1)

    #We pick out the median of each cluster as its value. We tried to pick out the average, but some outliers made it inconsistent 
    output = np.array([np.median(anchorBoxList[outputSmallestError == i], axis=0) for i in range(0,k)])

#Print the clustered bounding box priors, sorted  ascending by area
print(sorted(output*416, key=lambda tup: functools.reduce(mul, tup)))