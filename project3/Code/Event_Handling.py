import numpy as np
import cv2
import os 


def mouse_handler(event, x, y, flags, data):
    
    if event == cv2.EVENT_LBUTTONDOWN :
        cv2.circle(data['im'], (x,y),1, (0,0,255), 5, 16);
        cv2.imshow("Image", data['im']);
        if len(data['points']) <= 15 :
            data['points'].append([x,y])

                 
def get_points(im):
    
    data = {}
    data['im'] = im.copy()
    data['points'] = []
    
   
    cv2.imshow("Image",im)
    cv2.setMouseCallback("Image", mouse_handler, data)
    cv2.waitKey() 
    points = np.array(data['points'])
    
    return points


def transparent(Image):

    
    tmp = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,127,255,cv2.THRESH_BINARY)
    b, g, r = cv2.split(Image)
    rgba = [b,g,r, alpha]
    dst = cv2.merge(rgba,4)
    
    return dst

def crop_image(Image,Path,Counter):


    Image = cv2.imread(Image)
    Image = cv2.resize(Image,(1080,720))
    Image = cv2.medianBlur(Image,15)
    Image = cv2.bilateralFilter(Image,9,75,75) 
    Image = cv2.dilate(Image,np.ones((5,5),np.uint8),iterations =1)

    Points = get_points(Image)
    rect = cv2.boundingRect(Points)
    x,y,w,h = rect
    croped = Image[y:y+h-4,x:x+w-4].copy()
    
    Points = Points - Points.min(axis=0)
    
    mask1 = np.zeros(croped.shape[:2],np.uint8)
    cv2.drawContours(mask1,[Points],-1,(255,255,255),-1,cv2.LINE_AA)
    Cropped_Image1 = cv2.bitwise_and(croped,croped,mask=mask1)
    Final = transparent(Cropped_Image1)
    
    cv2.imwrite("%s/Cropped_Image%s.png"%(Path,Counter),Final)
    
def save_image(Image,Path,Color,Case):
    
    Image = Image
    if Case ==1:

        cv2.imwrite("%s/Train_Image%s.jpg"%(Path,Color),Image)


    if Case ==2:
       
        cv2.imwrite("%s/Test_Image%s.jpg"%(Path,Color),Image)

def crop_rectangle(Image,Path,Counter):
    
    Image = cv2.imread(Image)
    Image = cv2.resize(Image,(1440,960))
    Image = cv2.medianBlur(Image,15)
    Image = cv2.bilateralFilter(Image,9,75,75) 
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    Image = cv2.filter2D(Image, -1, kernel)
    Image = cv2.dilate(Image,np.ones((5,5),np.uint8),iterations =1)
    
    # Select ROI
    
    fromCenter = False
    r = cv2.selectROI("Image", Image, fromCenter)
     
    # Crop image
    imCrop = Image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
 
    # Display cropped image
    cv2.imwrite("%s/Cropped_Image%s.png"%(Path,Counter),imCrop)

    

