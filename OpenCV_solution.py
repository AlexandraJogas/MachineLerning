
########### Import necessary libraries ################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2


#######################################################
"""
1. Install OpenCV using the command: pip install opencv-python
2. If you get an error that “Visual Studio Redistributabel package is missing - install it.
Google it and you will get to the correct page.
3. The import command for OpenCV in python is: import cv2
"""
########################################################
"""
4. Load a color image using imread
http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_image_display/py_image_dis
play.html
"""
color_img = cv2.imread('Lenna.png')  


"""
5. Show the image and then keep the window open for 10 seconds (imshow, waitkey)
"""
cv2.imshow('image',color_img)  
cv2.waitKey(10000)     
cv2.destroyAllWindows() 


"""
6. Show the image and then keep the window open till a command is entered in the
window
"""
cv2.imshow('image',color_img)
cv2.waitKey()                
cv2.destroyAllWindows()       


"""
7. Print the value of the pixel (RGB) at location (100,100)
http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_core/py_basic_ops/py_basic_ops.ht
ml#basic-ops
"""
px = color_img[100,100]  
print(px)


"""
8. Print only the blue value of this pixel
"""
only_blue = color_img[100,100,0]   
print(only_blue)
#or
print(px[0])

"""
9. Access this pixel value and then modify it’s value using the “item” and “itemset” functions
"""
color_img.itemset((100,100,0),255)  
color_img.item((100,100,0))          

"""
10. Print the image shape, size and dtype values to the screen
"""
print(color_img.shape)   
print(color_img.size)    
print(color_img.dtype)   # uint8= unsigned integer8

"""
11. Choose a rectangluar area in the image, copy it’s values to a variable and then paste it
in another area in the pixel
"""
rect_area = color_img[120:180, 180:240]   
color_img[120:180, 240:300] = rect_area  



"""
12. Split the RGB image into three different images: R,G,B and show them. Use the “split”
function
"""
b,g,r = cv2.split(color_img)  


"""
13. Merge these values back to a RGB image using the “merge” function
"""
merged_img = cv2.merge((b,g,r))  


"""
14. Do the same operation now with NumPy indexing of the image
"""
b_idx = color_img[:,:,0]   
g_idx = color_img[:,:,1]
r_idx = color_img[:,:,2]



"""
15. Add borders around the image:
http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/copyMakeBorder/copyMakeBo
rder.html
Use two differnet flags: BORDER_REPLICATE 2. BORDER_CONSTANT
"""
red = [255,0,0]  # BGR                              
replicate = cv2.copyMakeBorder(color_img,10,10,10,10, cv2.BORDER_REPLICATE)  
constant  = cv2.copyMakeBorder(color_img,10,10,10,10, cv2.BORDER_CONSTANT, value =red)  

plt.imshow(color_img,'gray'),plt.title('Original')
plt.imshow(replicate,'gray'),plt.title('Replicate')
plt.imshow(constant, 'gray'),plt.title('Constant')
plt.show()
"""
16. Convert the RGB image into a Gray level image using “cvtColor” and “show”
"""
img2gray = cv2.cvtColor(color_img,cv2.COLOR_BGR2GRAY)  # cvtColor= convert color
plt.imshow(img2gray)  

"""
17. Create a binary image from the Gray image using “threshold” and show
"""                                    
ret1, binary_img = cv2.threshold(img2gray, 150, 255, cv2.THRESH_BINARY) 
plt.imshow(binary_img, cmap='grey')                                    


"""
18. Blur the image using “GaussianBlur” and show
"""
blurred_img = cv2.GaussianBlur(color_img,(7,7),3)  
plt.imshow(blurred_img)             

"""
19. Create a Gaussian kernel and filter using “filter2D”
"""
GaussKernel=cv2.getGaussianKernel((10,10),20) 
filt_2d = cv2.filter2D(color_img, -1, GaussKernel)   
plt.imshow(filt_2d)                        

"""
20. Create a three level pyramid using “pyrDown” and “pyrUp”
"""
pyrdown = cv2.pyrDown(color_img)   
plt.imshow(pyrdown)
pyrdown.shape   
plt.imshow(color_img)

pyrup = cv2.pyrUp(color_img)
plt.imshow(pyrup)
pyrdown.shape   # (1056,1056,1) tmuna  gdolya yoter

"""
21. Resize the image using “resize” and show
"""
resize = cv2.resize(color_img,(50,50))
plt.imshow(resize)

resize_2 = cv2.resize(color_img,(1000,1000))
plt.imshow(resize_2)


"""
22. Perform an Affine transformation using “warpAffine
"""
# nagdir matriza sivuv ve azaza
M = np.array([[0., .5, 1.],
              [.5, 0,  1.]]) 

aff_trans = cv2.warpAffine(color_img, M, (300,300)) 
plt.imshow(aff_trans)


"""
23. Create a rotation matrix from rotation angle using: getRotationMatrix2D and thenperform
the affine transofrm using warpAffine
"""
angle = 74
angle_in_rad = angle*np.pi/180
rotation_mat = cv2.getRotationMatrix2D((50,50), angle_in_rad, 0.5)  
aff_trans_2 = cv2.warpAffine(color_img, rotation_mat, (300,300))    
plt.imshow(aff_trans_2)

