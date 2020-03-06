import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

def Parameters(samples):

    mu = np.mean(samples)
    sigma = np.std(samples)

    return mu,sigma
    

def Gaussian1D(x,mu,sigma):

    A = 1/(sigma*np.sqrt(2*math.pi))
    Z = A*np.exp(-(x-mu)**2/(2.*sigma**2))

    return Z



muG = 192.18119099491648
stdG = 33.13959266379306

muR = 244.8884335154827
stdR = 15.988621099080165


muY = 213.25615384615384
stdY = 17.8243214874831

cap = cv2.VideoCapture('../Data/detectbuoy.avi')

if (cap.isOpened()== False): 
   print("Error opening video stream or file")

count = 0
while(cap.isOpened()):
   
   
    ret, frame = cap.read()
    
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    OI = frame
    I = cv2.medianBlur(OI,15)
    I = cv2.bilateralFilter(I,9,75,75)
    
    I = cv2.dilate(I,np.ones((5,5),np.uint8),iterations =1)

    Ig = ((I[:,:,1] - I[:,:,2]))
    
    Ir = I[:,:,0] 
    Iy = (I[:,:,1] + I[:,:,0] - I[:,:,2])

    Probg = Gaussian1D(Ig,muG,stdG)
    Probr = Gaussian1D(Ir,muR,stdR)
    Proby = Gaussian1D(Iy,muY,stdY)
    

    
    Probg = Probg/np.amax(Probg)
    Probr = Probr/np.amax(Probr)
    Proby = Proby/np.amax(Proby)
    
    Green = (Probg>= 0.0008) 
    Red = (Probr>=0.68) 
    
    Final =  (Red | Green)
    
    plt.figure(1),plt.clf()
    plt.imshow(OI)
    plt.contour(Final)
    
    plt.pause(0.00000000003)
    
    
    
cap.release()
cv2.destroyAllWindows()




