import cv2
import numpy as np
import math
from matplotlib import pyplot as plt



def hist(Image,block=False):
    image = cv2.imread(Image)
    
    histogram1=cv2.calcHist([image], [0], None, [256], [0,256])
    histogram2=cv2.calcHist([image], [1], None, [256], [0,256])
    histogram3=cv2.calcHist([image], [2], None, [256], [0,256])
        
    return histogram1,histogram2,histogram3    
        
i = 1
n = 7   #Number of Cropped Images.
s1g = 0
s2g = 0
s3g = 0
while(i <= 25):

    
    h1,h2,h3 = hist('../Data/Frames/Green/Training/Cropped/Cropped_Image{}.png'.format(i))
      
    s1g = s1g + h1
    s2g = s2g + h2
    s3g = s3g + h3
    i = i+4
    
        
#Averaging the Histogram.
    
s1g = s1g/n
s2g = s2g/n
s3g = s3g/n



plt.title('Green Buoy')
plt.subplot(3,1,1) 
plt.plot(s1g,color = 'b')

plt.subplot(3,1,2) 
plt.plot(s2g,color = 'g')

plt.subplot(3,1,3) 
plt.plot(s3g,color = 'r')

plt.show(block = False) 
    
    
j = 1

s1r = 0
s2r = 0
s3r = 0
s1y = 0
s2y = 0
s3y = 0
while(j <= 121):

    h1r,h2r,h3r = hist('../Data/Frames/Red/Training/Cropped/Cropped_Image{}.png'.format(j))
    h1y,h2y,h3y = hist('../Data/Frames/Yellow/Training/Cropped/Cropped_Image{}.png'.format(j))
    s1r = s1r + h1r
    s2r = s2r + h2r
    s3r = s3r + h3r
    s1y = s1y + h1y
    s2y = s2y + h2y
    s3y = s3y + h3y
    
    j = j+20
        
    
s1r = s1r/n
s2r = s2r/n
s3r = s3r/n
s1y = s1y/n
s2y = s2y/n
s3y = s3y/n

plt.figure(2)
plt.subplot(3,1,1) 
plt.plot(s1r,color = 'b')

plt.subplot(3,1,2) 
plt.plot(s2r,color = 'g')

plt.subplot(3,1,3) 
plt.plot(s3r,color = 'r')
plt.show(block = False) 
    

plt.figure(3)
plt.subplot(3,1,1) 
plt.plot(s1y,color = 'b')

plt.subplot(3,1,2) 
plt.plot(s2y,color = 'g')

plt.subplot(3,1,3) 
plt.plot(s3y,color = 'r')
plt.show(block = True)    



