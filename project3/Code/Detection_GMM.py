import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
# from Green_3Gaussian_1D import muGreen
# from Red_3Gaussian_1D import muRed
# from Yellow_3Gaussian_1D import muYellow
import matplotlib.image as mpimg

def Parameters(samples):

    mu = np.mean(samples)
    sigma = np.std(samples)

    return mu,sigma
    

def Gaussian1D(x,mu,sigma):

    A = 1/(sigma*np.sqrt(2*math.pi))
    Z = A*np.exp(-(x-mu)**2/(2.*sigma**2))

    return Z



# The Mean and Sigma values obtained from the 3Gaussian-1D (EM Method) is used here to
# detect the color buoys and draw the contours


mu1 = [135.22750891,215.88834476,130.81198099]
mu2=[204.1577144,232.0026915,205.21395644]
mu3=[177.20187941, 247.2919645,181.65907]

sigma1=[1523.35575922,54.653475,1869.28539]
sigma2=[506.260641,236.439907,549.204234]
sigma3=[1036.3296992,59.3066744,1282.98152719]

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

    # Probg = Gaussian1D(Ig,muG,stdG)
    # Probr = Gaussian1D(Ir,muR,stdR)
    # Proby = Gaussian1D(Iy,muY,stdY)

    Probg = 0
    Probr = 0
    Proby = 0

    # Multivariate Normal distribution function
    for i in range(0, (len(mu1))):
        Proby = Gaussian1D((Iy), mu1[i], sigma1[i]) + Proby

    for j in range(0, (len(mu2))):
        Probg = Gaussian1D((Iy), mu2[i], sigma2[i]) + Probg

    for k in range(1, (len(mu3))):
        Probr = Gaussian1D((Ir), mu3[k], sigma3[k]) + Probr

    
    Probg = Probg/np.amax(Probg)
    Probr = Probr/np.amax(Probr)
    Proby = Proby/np.amax(Proby)
    print("Probg",Probg)
    print("Probr",Probr)
    Green = (Probg<0.966)
    Red = (Probr>=0.885)
    
    Final =  (Red | Green)
    
    
    plt.figure(1),plt.clf()
    plt.imshow(OI)
    plt.contour(Final)
    plt.pause(0.00000000003)
    
    
cap.release()
cv2.destroyAllWindows()




