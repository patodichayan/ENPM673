import numpy as np 
import matplotlib.pyplot as plt
import pickle
import random



##opening of the file

f = open('data3_new.pkl','rb') 
data = pickle.load(f)
f.close()
da = np.asarray(data)
X = da[:,0]
Y = da[:,1]

#initialisation of variables

cnt = 0
best_fit = 0
slope = 0
intercept = 0
iterations = 1
variance = 0
inlier = 0
outlier = 0
l = len(da)
p = 0.9



while iterations > cnt : 
	xyrand1 = random.choice(da) ## generation of random variables
	xyrand2 = random.choice(da)
	rand_data = np.array( (xyrand1, xyrand2) ) 
	
	##line fitting and calculation of slope/intercept
	m = (rand_data[1,1] - rand_data[0,1])/(rand_data[1,0]-rand_data[0,0]) 
	c = rand_data[1,1] - (m*rand_data[1,0])

	distance = []
	inlier = 0 #outlier and inlier initialisation at each iteration in order to compare with prev 			    value
	outlier = 0
	

	for i in range(len(da)) : 
	
		#calculation of distance of a point from the fitted line
		D = ( c - Y[i] + (m*X[i]) )/(np.sqrt( (m**2)+1) )
		
		#distance array is used to compute the distance of each point from the fitted line. It 			is done by first computing the variance of the distane array in order to know 	how the 		gradient occurs for a particular pair of chosen points. This variance is used to compute 			the required threshold for outlier rejection.
		distance = np.append(distance, D)

	variance = np.var(distance)
	thres = np.sqrt(3.84*variance)

	for i in range(l) : 
		if(distance[i] < thres) : #inilier is identified if the points distance is less than the threshold given.
			inlier += 1 
		else : 
			outlier+= 1

	e = (1 - (float(inlier))/l) 
	iterations = int(np.log10(1-p))/(np.log10(1-((1-e)**2)))
	cnt+=1

	if (best_fit < inlier) :
		#the variable best_fit is used to identify whether the chosen set of points in the 			current iteration generate the best line that has the maximum number of inlier to 			outlier ratio. The best_fit variable stores the previous inlier value and is compared to 			the previous value. If the current inliers are more then it means the current model is 			more efficient than the prev one.
		best_fit = inlier
		intercept = c
		slope = m
	else :
		intercept = intercept
		best_fit = best_fit
		slope = slope

y_predict = (slope*X) + intercept
plt.figure(1)
#plt.subplot(132)
plt.plot(X,Y,'ro')
plt.plot(X, y_predict, color = 'k', linewidth = 2)
plt.axis([-150,150,-100,100])
plt.show()




