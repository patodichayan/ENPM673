import cv2
import numpy as np
import matplotlib.pyplot as plt

frame = cv2.imread('../Data/frame1.jpg')
#cv2.imshow('',frame)
#cv2.waitKey()

# Convert BGR to HSV
#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

OI = frame

frame = cv2.medianBlur(frame,15)
frame = cv2.bilateralFilter(frame,9,75,75) 
frame = cv2.dilate(frame,np.ones((5,5),np.uint8),iterations =1)

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# define range of blue color in HSV
lower_green = np.array([45,100,70])
upper_green = np.array([62,255,255])

lower_red = np.array([0,115,30])
upper_red = np.array([18,255,255])

lower_yellow = np.array([30,111,30])
upper_yellow = np.array([40,255,255])

# Threshold the HSV image to get only blue colors

mask1 = cv2.inRange(hsv, lower_green, upper_green)
mask2 = cv2.inRange(hsv, lower_red, upper_red)
mask3 = cv2.inRange(hsv, lower_yellow, upper_yellow)

Final = mask1 | mask2 | mask3

cv2.imshow('',Final)

cv2.waitKey()
cont,h= cv2.findContours(Final,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
Z = cv2.drawContours(OI,cont,-1,(0,255,0),2) 

cv2.imshow('img',Z)

cv2.waitKey()


