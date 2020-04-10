'''
生成旋转、平移和缩放的图片
'''
import numpy as np #1
import argparse #2
import imutils #3
import cv2 #4
 
 
image = cv2.imread("einstein.jpg") #8
cv2.imshow("Original", image) #9
 
(h, w) = image.shape[:2] #10
center = (w // 2, h // 2) #11
# 缩放
scaled = cv2.resize(image, (int(w/2), int(h/2)))
cv2.imwrite('img_sca.jpg', scaled)
# 平移
imgInfo = image.shape
height = imgInfo[0]
width = imgInfo[1]
mode = imgInfo[2]
dst = np.zeros(imgInfo, np.uint8)

for i in range( height ):
    for j in range( width - 100 ):
        dst[i, j + 100] = image[i, j]
cv2.imwrite('img_tra.jpg', dst)
# 旋转
M = cv2.getRotationMatrix2D(center, 45, 1.0) #12
rotated = cv2.warpAffine(image, M, (w, h)) #13
cv2.imshow("Rotated by 45 Degrees", rotated) #14
cv2.imwrite('img45.jpg', rotated)
 

