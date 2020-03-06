


# Write python code (using numpy and matplotlib) to visualize the geometric interpretation of the eigen-values and the covariance matrix.

import numpy as np
import matplotlib.pyplot as plt
import pickle

# Extracting the Data

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

Z1 = np.stack((X1,Y1),axis=0)
Z2 = np.stack((X2,Y2),axis=0)
Z3 = np.stack((X3,Y3),axis=0)

# Covariance Matrices.

D1 = np.cov(Z1)
D2 = np.cov(Z2)
D3 = np.cov(Z3)

# Eigen-Values & Eigen-Vectors.

EigenValue1,EigenVector1 = np.linalg.eig(D1)
EigenValue2,EigenVector2 = np.linalg.eig(D2)
EigenValue3,EigenVector3 = np.linalg.eig(D3)

#Geometrical Interpretation.

plt.figure(1)
plt.plot(X1,Y1,'ro')
plt.axis([-150,150,-100,100])
plt.title('for data1_new.pkl')                                                   
plt.quiver(np.mean(X1),np.mean(Y1),np.sqrt(EigenValue1)*EigenVector1[0,:],np.sqrt(EigenValue1)*EigenVector1[1,:],scale_units = 'xy', scale=0.5)

plt.figure(2)
plt.plot(X2,Y2,'ro')
plt.axis([-150,150,-100,100])
plt.title('for data2_new.pkl')
plt.quiver(np.mean(X2),np.mean(Y2),np.sqrt(EigenValue2)*EigenVector2[0,:],np.sqrt(EigenValue2)*EigenVector2[1,:],scale_units = 'xy', scale=0.5)

plt.figure(3)
plt.plot(X3,Y3,'ro')
plt.axis([-150,150,-100,100])
plt.title('for data3_new.pkl')
plt.quiver(np.mean(X3),np.mean(Y3),np.sqrt(EigenValue3)*EigenVector3[0,:],np.sqrt(EigenValue3)*EigenVector3[1,:],scale_units = 'xy', scale=0.5)
plt.show()



