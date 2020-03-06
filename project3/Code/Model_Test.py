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


muG = 221.2364824777658
stdG = 33.13959266379306

muR = 244.91235431235
stdR = 15.905603947367796

muY = 213.25615384615384
stdY = 17.8243214874831

i = 29

#For Test Images in Green Folder.

while(i<=39):

    frame = cv2.imread('../Data/Frames/Green/Test/Test_ImageGreen{}.jpg'.format(i))    
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
    
    i = i + 2
    Final =  (Red | Green)
    
    plt.figure(1)
    plt.imshow(OI)
    plt.contour(Final)
    
    plt.show()
    

#For Test Images in Red Folder.
j = 141

while(j<=191):

    frame = cv2.imread('../Data/Frames/Red/Test/Test_ImageRed{}.jpg'.format(j))    
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
    
    j = j + 10
    Final =  (Red | Green)
    
    plt.figure(2)
    plt.imshow(OI)
    plt.contour(Final)
    
    plt.show()

#For Test Images in Yellow Folder.
g = 141

while(g<=191):

    frame = cv2.imread('../Data/Frames/Yellow/Test/Test_ImageYellow{}.jpg'.format(g))    
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
    
    g = g + 10
    Final =  (Red | Green)
    
    plt.figure(3)
    plt.imshow(OI)
    plt.contour(Final)
    
    plt.show()

    
    




