
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import pickle
import scipy.stats as st
from numpy import array
from numpy.linalg import eig
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn import linear_model, datasets

#Our given data
f1 = open('data1_py2.pkl','rb')
f2 = open('data2_py2.pkl','rb')
f3 = open('data3_py2.pkl','rb')

#loading the data from the file
data1 = pickle.load(f1)
data2 = pickle.load(f2)
data3 = pickle.load(f3)

#closing the files
f1.close()
f2.close()
f3.close()

#data is stored in an array for each datasets
dat1 = np.asarray(data1)
dat2 = np.asarray(data2)
dat3 = np.asarray(data3)

#X and Y values represent each values of data array in its first and second position
X1= dat1[:,0]
Y1= dat1[:,1]

X2 = dat2[:,0]
Y2 = dat2[:,1]

X3 = dat3[:,0]
Y3 = dat3[:,1]

#Obtaining the X values and conveting it into a matrix
for i in range (0,200):
    a1=np.array([X1[i],1])
    e1=a1.tolist()
    A1=np.matrix(e1)
    
    f1=np.array([Y1[i]])
    g1=f1.tolist()
    b1=np.matrix(g1)

#SVD calculation
#U-represents the Unitary Matrix
#s-Rectangular Diagonal Matrix
#V-Conjugate transpose of the n by n unitary matrix 

U1, s1, V1 = np.linalg.svd(A1)     

#Slope and intercept calculation from U, s and V matrix       
r1 = [0.0, 0.0]
r1 += (1/s1[0]) * (U1[:,0].T*b1) * (V1[:,0].T) 
#r1 gives the slope and intercept values

#Similarly method done for Dataset2
for i in range (0,200):
    a2=np.array([X2[i],1])
    e2=a2.tolist()
    A2=np.matrix(e2)
    
    f2=np.array([Y2[i]])
    g2=f2.tolist()
    b2=np.matrix(g2)


U2, s2, V2 = np.linalg.svd(A2)
       
r2 = [0.0, 0.0]
r2 += (1/s2[0]) * (U2[:,0].T*b2) * (V2[:,0].T) 

#Similarly method done for Dataset2
for i in range (0,200):
    a3=np.array([X3[i],1])
    e3=a3.tolist()
    A3=np.matrix(e3)
    
    f3=np.array([Y3[i]])
    g3=f3.tolist()
    b3=np.matrix(g3)


U3, s3, V3 = np.linalg.svd(A3)
       
r3 = [0.0, 0.0]
r3 += (1/s3[0]) * (U3[:,0].T*b3) * (V3[:,0].T) 

#Slope and intercept determination
m1=r1.item(0)
c1=r1.item(1)
m2=r2.item(0)
c2=r2.item(1)
m3=r3.item(0)
c3=r3.item(1)

#Line fitting-x and y values determination
Reg_x1 = [-100,100]
Reg_y1 = [c1+m1*-100 , c1+m1*100]

Reg_x2 = [-100,100]
Reg_y2 = [c2+m2*-100 , c2+m2*100]

Reg_x3 = [-100,100]
Reg_y3 = [c3+m3*-100 , c3+m3*100]


'''
Plots the linear regression line,
Or really any array as a line
'''

#Plotting figure
#Figure1
plt.figure(1)
plt.subplot(131)
plt.plot(X1,Y1,'ro')
plt.plot(Reg_x1,Reg_y1, color='g', linewidth=3)


#Figure2
plt.figure(1)
plt.subplot(132)
plt.plot(X2,Y2,'ro')
plt.plot(Reg_x2,Reg_y2, color='g', linewidth=3)
plt.axis([-150,150,-100,100])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('SVD for Data-1,Data-2 and Data-3\n')

#Figure3
plt.figure(1)
plt.subplot(133)
plt.plot(X3,Y3,'ro')
plt.plot(Reg_x3,Reg_y3, color='g', linewidth=3)


plt.show()



















