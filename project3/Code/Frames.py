import numpy as np
import cv2
import os
from Event_Handling import *


os.makedirs('../Data/Frames/OriginalFrames')
os.makedirs('../Data/Frames/Green/Training/Cropped')
os.makedirs('../Data/Frames/Green/Test')
os.makedirs('../Data/Frames/Red/Training/Cropped')
os.makedirs('../Data/Frames/Red/Test')
os.makedirs('../Data/Frames/Yellow/Training/Cropped')
os.makedirs('../Data/Frames/Yellow/Test')


cap = cv2.VideoCapture('detectbuoy.avi')


if (cap.isOpened()== False): 
   print("Error opening video stream or file")

count = 0
while(cap.isOpened()):
      
   ret, frame = cap.read()
   
   if ret ==True:

           count += 1 
           cv2.imwrite(os.path.join('../Data/Frames/OriginalFrames',"frame{:d}.jpg".format(count)),frame) 
                
                               
   if ret ==False:
       break

flag1 = 1

while(flag1<=42):
    
    im1 = cv2.imread(os.path.join('../Data/Frames/OriginalFrames',"frame{}.jpg".format(flag1)))
    
    
        
    if flag1 <=28:
        
        save_image(im1,"../Data/Frames/Green/Training","Green{}".format(flag1),1)
        
          
    else:
        
        save_image(im1,"../Data/Frames/Green/Test","Green{}".format(flag1),2)
        
    flag1 = flag1 + 2

flag2 = 1

while(flag2<=200):

    im2 = cv2.imread(os.path.join('../Data/Frames/OriginalFrames',"frame{}.jpg".format(flag2)))
    
    if flag2 <=140:
        
        save_image(im2,"../Data/Frames/Red/Training","Red{}".format(flag2),1)
        save_image(im2,"../Data/Frames/Yellow/Training","Yellow{}".format(flag2),1)
        
       

    else:
        
        save_image(im2,"../Data/Frames/Red/Test","Red{}".format(flag2),2)
        save_image(im2,"../Data/Frames/Yellow/Test","Yellow{}".format(flag2),2)
        
    flag2 = flag2 + 10


counter1 = 1  
counter2 = 1  
while(counter1<=28):

    if counter1 <= 28:
            
            print('Create a rectangle on Green Buoy . Press Enter. ')
            crop_rectangle("../Data/Frames/Green/Training/Train_ImageGreen{}.jpg".format(counter1),"../Data/Frames/Green/Training/Cropped",counter1)
            counter1 = counter1 + 4
            
    if counter2 <= 140:
            
            print('Create a rectangle on Red Buoy. Press Enter.')
            crop_rectangle("../Data/Frames/Red/Training/Train_ImageRed{}.jpg".format(counter2),"../Data/Frames/Red/Training/Cropped",counter2)
            print('Create a rectangle on Yellow Buoy. Press Enter.')          
            crop_rectangle("../Data/Frames/Yellow/Training/Train_ImageYellow{}.jpg".format(counter2),"../Data/Frames/Yellow/Training/Cropped",counter2)
            
            counter2 = counter2 + 20 
      


cap.release()
cv2.destroyAllWindows()

