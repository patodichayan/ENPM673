


#  Write code to fit to these datasets a line using Least Squares using 
#  b.) The orthogonal distances. 



import numpy as np
import matplotlib.pyplot as plt
import pickle


# Extracting The Data.

f1 = open('data1_new.pkl','rb')
f2 = open('data2_new.pkl','rb')
f3 = open('data3_new.pkl','rb')

data1 = pickle.load(f1)
data2 = pickle.load(f2)
data3 = pickle.load(f3)

f1.close()
f2.close()
f3.close()

dat1 = np.asarray(data1)
dat2 = np.asarray(data2)
dat3 = np.asarray(data3)

X1 = dat1[:,0]
Y1 = dat1[:,1]

X2 = dat2[:,0]
Y2 = dat2[:,1]

X3 = dat3[:,0]
Y3 = dat3[:,1]

n = 1,2,3;

# Creating a Function for slope & intercept.

def slope_intercept(Xn,Yn):

	Ysq = Yn**2	
	Ysum = sum(Ysq)
	Xsq = Xn**2	
	Xsum = sum(Xsq)

	Num = Ysum-len(Yn)*(np.mean(Yn)**2) - Xsum-len(Xn)*(np.mean(Xn)**2)
	Den = np.mean(Xn)*np.mean(Yn)*len(Xn) - sum(Xn*Yn)
	 
	B = 0.5*(Num/Den)
	m = np.sqrt(np.square(B) + 1) - B
	b = np.mean(Yn) - m*np.mean(Xn)

	return m,b


# Plotting.

m,b = slope_intercept(X1,Y1)
best_fit1 = [(m*x)+b for x in X1]
plt.figure(1)
plt.title('Line Fitting')
plt.subplot(311)
plt.plot(X1,Y1,'ro')
plt.plot(X1,best_fit1)
plt.axis([-150,150,-100,100])

m,b = slope_intercept(X2,Y2)
best_fit2 = [(m*x)+b for x in X2]
plt.figure(1)
plt.subplot(312)
plt.plot(X2,Y2,'ro')
plt.plot(X2,best_fit2)
plt.axis([-150,150,-100,100])

m,b = slope_intercept(X3,Y3)
best_fit3 = [(m*x)+b for x in X3]
plt.figure(1)
plt.subplot(313)
plt.plot(X3,Y3,'ro')
plt.plot(X3,best_fit3)
plt.axis([-150,150,-100,100])

plt.show()
	     	
