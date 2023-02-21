
"""
Spyder Editor

This is a temporary script file.
"""

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
color_img = cv2.imread('Lenna.png')  # naumpy array pixel shel tmuna (512X512X3) 3=aruzim =channels= BGR blue,red,green


"""
5. Show the image and then keep the window open for 10 seconds (imshow, waitkey)
"""
cv2.imshow('image',color_img)  # liftoah tmuna
cv2.waitKey(10000)      # bemaalah riza tiftah 10 sec ve tisgor,  10000 milisec=10 sec, im =0 tehake ad she elhaz al kol kaftor
cv2.destroyAllWindows() # lisgor halonit shel image she patahta


"""
6. Show the image and then keep the window open till a command is entered in the
window
"""
cv2.imshow('image',color_img)
cv2.waitKey()                 # liftoah halon ad she le lohzim euze she kaftor, 0= gam efshar lihtov
cv2.destroyAllWindows()       # lisgor halonit shel image she patahta


"""
7. Print the value of the pixel (RGB) at location (100,100)
http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_core/py_basic_ops/py_basic_ops.ht
ml#basic-ops
"""
px = color_img[100,100]  # ze numpy array, mazig ereh be makom (100,100) BGR [255,60,100]= ze gavan ve pixel 100
print(px)


"""
8. Print only the blue value of this pixel
"""
only_blue = color_img[100,100,0]    # lageshet le ereh kahol =255, 0=blue, 1=green, 2= red
print(only_blue)
#or
print(px[0])

"""
9. Access this pixel value and then modify it’s value using the “item” and “itemset” functions
"""
color_img.itemset((100,100,0),255)   # be makom (100,100,0)  leaziv ereh 255, laasot set le item
color_img.item((100,100,0))          # leahzir item be makom (100,100,0)

"""
10. Print the image shape, size and dtype values to the screen
"""
print(color_img.shape)   # efshar leadpis shape, size, dtype kmo be numpy array  (512,512,3)
print(color_img.size)    # (512,512,3)= 786.432  kama pixel or arahim mahilya tmuna
print(color_img.dtype)   # uint8= unsigned integer8

"""
11. Choose a rectangluar area in the image, copy it’s values to a variable and then paste it
in another area in the pixel
"""
rect_area = color_img[120:180, 180:240]   # lehozia helek me tmuna
color_img[120:180, 240:300] = rect_area   # ve lasim be koordinata aheret



"""
12. Split the RGB image into three different images: R,G,B and show them. Use the “split”
function
"""
b,g,r = cv2.split(color_img)  # lakahat image ve lehalek le 3 tmunot nifradot


"""
13. Merge these values back to a RGB image using the “merge” function
"""
merged_img = cv2.merge((b,g,r))  # lakahat 3 tmunot ve lemazeg le tmuna ahat


"""
14. Do the same operation now with NumPy indexing of the image
"""
b_idx = color_img[:,:,0]   # laasot split be ezrat numpy array, lakahat kol shurot, kol amudot ve channel blue
g_idx = color_img[:,:,1]
r_idx = color_img[:,:,2]



"""
15. Add borders around the image:
http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/copyMakeBorder/copyMakeBo
rder.html
Use two differnet flags: BORDER_REPLICATE 2. BORDER_CONSTANT
"""
red = [255,0,0]  # BGR                                # BORDER_REPLICATE= meshahpel pixels be zdadim ose oto davar, pixels krovim
replicate = cv2.copyMakeBorder(color_img,10,10,10,10, cv2.BORDER_REPLICATE) # godel border 10,10,10,10=pixels bezdadim, povtorit, rastyanut 
constant  = cv2.copyMakeBorder(color_img,10,10,10,10, cv2.BORDER_CONSTANT, value =red)  # border im ereh kavua, ramka krasnaya budet

plt.imshow(color_img,'gray'),plt.title('Original')
plt.imshow(replicate,'gray'),plt.title('Replicate')
plt.imshow(constant, 'gray'),plt.title('Constant')
plt.show()
"""
16. Convert the RGB image into a Gray level image using “cvtColor” and “show”
"""
img2gray = cv2.cvtColor(color_img,cv2.COLOR_BGR2GRAY)  # cvtColor= convert color, leyazer tmuna gray
plt.imshow(img2gray)  # roim green ve le gray ki le katavnu plt.imshow(img2gray,cmap='gray')


"""
17. Create a binary image from the Gray image using “threshold” and show
"""                                       # anu rozim THRESH_BINARY 2 option: black/white 0/255
ret1, binary_img = cv2.threshold(img2gray, 150, 255, cv2.THRESH_BINARY) # 150 krai=saf eifo iani hoteh= hituh
plt.imshow(binary_img, cmap='grey')                                     # 255 ma ihie ereh gavoa = ma ani mazeva be ereh ahi gavoa


"""
18. Blur the image using “GaussianBlur” and show
"""
blurred_img = cv2.GaussianBlur(color_img,(7,7),3)  # osim convolution shel tmuna keasher (7,7)=filtr=karnel zura gausian, 3=sigma gausian
plt.imshow(blurred_img)             # filter Gaussiany ose tishtush le tmuna, sigma gausiian ma ihie efresh arahim bein emza le misasaviv

"""
19. Create a Gaussian kernel and filter using “filter2D”
"""
GaussKernel=cv2.getGaussianKernel((10,10),20)  # zura shel filtr gaussian, naziv filter gaussiany le filter2D, 10X10 godel filter, 20=sigma std
filt_2d = cv2.filter2D(color_img, -1, GaussKernel)   # filter du meimadi= filter2D, -1= input shave le output be sug tmuna
plt.imshow(filt_2d)                         # 20= sigma kovea distribution shel arahim

"""
20. Create a three level pyramid using “pyrDown” and “pyrUp”
"""
pyrdown = cv2.pyrDown(color_img)   # pyramida= kshe rozim lehistakel al scale tmunot be bat ahat
plt.imshow(pyrdown)
pyrdown.shape   # (256,256,1) tmuna ktana yoter

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
              [.5, 0,  1.]])  # laasot sivuv, ahaza, matiha: matriza (2,2) sivuv, (2,1) azaza
# laasot sivuv, ahaza, matiha le tmuna
aff_trans = cv2.warpAffine(color_img, M, (300,300))  # ze resize (300,300) kama pixels ani roza be tozaa
plt.imshow(aff_trans)


"""
23. Create a rotation matrix from rotation angle using: getRotationMatrix2D and thenperform
the affine transofrm using warpAffine
"""
angle = 74
angle_in_rad = angle*np.pi/180
rotation_mat = cv2.getRotationMatrix2D((50,50), angle_in_rad, 0.5)  # nagdir matriza sivuv ve azaza
aff_trans_2 = cv2.warpAffine(color_img, rotation_mat, (300,300))    # laasot sivuv, ahaza, matiha le tmuna
plt.imshow(aff_trans_2)

