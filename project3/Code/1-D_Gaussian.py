import numpy as np
import cv2
import math
from matplotlib import pyplot as plt


def Parameters(samples):

    mu = np.mean(samples)
    sigma = np.std(samples)

    return mu,sigma
    

def Gaussian1D(x,mu,sigma):

    A = 1/(sigma*np.sqrt(2*math.pi))
    Z = A*np.exp(-(x-mu)**2/(2.*sigma**2))

    return Z

i = 1

while (i<=28):
     GreenI = cv2.imread('../Data/Frames/Green/Training/Cropped/Cropped_Image{}.png'.format(i))
     GChannel = (GreenI[:,:,1])
     GChannel = np.concatenate(GChannel)
     
     i = i+4
     

mug,sigmag = Parameters(GChannel)    
Fg = Gaussian1D(GChannel,mug,sigmag)

plt.figure(1)
plt.stem(GChannel,Fg,'g',markerfmt='go')
plt.show(block = False)

j = 1

while (j<=121):
     RedI = cv2.imread('../Data/Frames/Red/Training/Cropped/Cropped_Image{}.png'.format(j))
     RChannel = ( RedI[:,:,2] )
     RChannel = np.concatenate(RChannel)
     
     
     YellowI = cv2.imread('../Data/Frames/Yellow/Training/Cropped/Cropped_Image{}.png'.format(j))
     YChannel = (YellowI[:,:,1] + YellowI[:,:,2])
     
     YChannel = np.concatenate(YChannel)
     
     j = j+20


mur,sigmar = Parameters(RChannel)

muy,sigmay = Parameters(YChannel)



Fr = Gaussian1D(RChannel,mur,sigmar)
Fy = Gaussian1D(YChannel,muy,sigmay)


print("Meanr",mur,'Stdr',sigmar)
print("Meang",mug,'Stdg',sigmag)
print("Meany",muy,'Stdy',sigmay)

plt.figure(2)
plt.stem(RChannel,Fr,'r',markerfmt='ro')
plt.show(block = False)

plt.figure(3)
plt.stem(YChannel,Fy,'y',markerfmt='yo')

plt.show()





