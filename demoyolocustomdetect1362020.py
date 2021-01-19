from darkflow.net.build import TFNet
from matplotlib import pyplot as plt
import cv2
##%matplotlib inline
import numpy as np # Import Numpy library
import pandas as pd # Import Pandas library

import sys # Enables the passing of arguments
import webcolors as wc
import math
from scipy import ndimage


dobjlstwor=[]
dobjlstwr=[]
font = cv2.FONT_HERSHEY_SIMPLEX
options = {"model": '/content/drive/My Drive/darkflow/cfg/tiny-yolo-voc-10c.cfg', "load":125, "threshold": 0.1}

tfnet = TFNet(options)

def object_detection(imgcv,mask,dobjlst):
    result = tfnet.return_predict(imgcv)
##print(result)
    print(type(result))
    for re in result:
        center_x=((re['topleft']['x']+re['bottomright']['x'])//2)-10
        center_y=((re['topleft']['y']+re['bottomright']['y'])//2)
        b,g,r=imgcv[center_y,center_x] # center_x and center_y ahi calculate thay pachi line lakhvi
        print(r,g,b)
        requested_colour = (r,g,b)
        actual_name, closest_name = get_colour_name(requested_colour)
        #ahi je closet color ave te aapde consider karvo tene aapda detected object sathe merge kari devo 
    ##    print ("Actual colour name:", actual_name, ", closest colour name:", closest_name)
        
        re['label']=closest_name+" "+re['label']
        print(re['label'])
        dobjlst.append(re['label'])
        cv2.rectangle(imgcv,(re['topleft']['x'],re['topleft']['y']),(re['bottomright']['x'],re['bottomright']['y']),(0,0,255),1)
        cv2.rectangle(mask, (re['topleft']['x'],re['topleft']['y']),(re['bottomright']['x'],re['bottomright']['y']), (0,0,0), thickness=-1)
##        cv2.putText(imgcv,re['label'],(center_x,center_y), font, 1, (255,0,0), 2, cv2.LINE_AA)

    return imgcv,mask,dobjlst


def closest_colour(requested_colour):
    min_colours = {}
    for key, name in wc.css3_hex_to_names.items():
        r_c, g_c, b_c = wc.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = wc.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name
## color detetcion mate preprocessing step end



##imgcv = cv2.imread("./sample_img/sample_dog.jpg")




imgcv = cv2.imread("/content/drive/My Drive/positiveimageset/157.jpg")
mask=np.ones_like(imgcv, np.uint8)*255
##print(imgcv[808,958]) # here height first then width ex. img(height,width) so img(y,x)

imgcv,mask,dobjlstwor= object_detection(imgcv,mask,dobjlstwor)
print("before rotation detected object",dobjlstwor)
cv2.imwrite("mask.png", mask)
mask = cv2.imread('mask.png',0)
cv2.imwrite("withoutrotation.png",imgcv)
res = cv2.bitwise_and(imgcv,imgcv,mask = mask)
cv2.imwrite("withoutrotationmasking.png",res)
imgcv=cv2.imread("withoutrotationmasking.png")
plt.imshow(res)
plt.show()
##cv2.imshow("without rotation masking",res)
##rotation code
##imgcv=res
img_gray = cv2.cvtColor(imgcv, cv2.COLOR_BGR2GRAY)
img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)

angles = []

for x1, y1, x2, y2 in lines[0]:
    cv2.line(imgcv, (x1, y1), (x2, y2), (255, 0, 0), 3)
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    angles.append(angle)

median_angle = np.median(angles)
img_rotated = ndimage.rotate(imgcv, median_angle)

img_rotated,mask,dobjlstwr= object_detection(img_rotated,mask,dobjlstwr)
print("after rotation detected object",dobjlstwr)
cv2.imwrite("withrotation.png",img_rotated)
plt.imshow(img_rotated)
plt.show()
##cv2.imshow("withrotation image",img_rotated)
##rotation code end
print("detected objects",dobjlstwr)
##   print(type(r))
##    print(r['label'])
##    print(r)
##cv2.imshow('output001',imgcv)

##cv2.waitKey(0)
##cv2.destroyAllWindows()
##from imageai.Detection import ObjectDetection
##
##detector = ObjectDetection()
##
##model_path = "./models/yolo-tiny.h5"
##input_path = "./input/carimg1.png"
##output_path = "./output/carimg1.png"
##
##detector.setModelTypeAsTinyYOLOv3()
##detector.setModelPath(model_path)
##detector.loadModel()
##detection = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path)
##
##for eachItem in detection:
##    print(eachItem["name"] , " : ", eachItem["percentage_probability"])
