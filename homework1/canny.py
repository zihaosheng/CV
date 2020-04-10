"""
cv2.Canny(image,            # 输入原图（必须为单通道图）
          threshold1, 
          threshold2,       # 较大的阈值2用于检测图像中明显的边缘
          [, edges[, 
          apertureSize[,    # apertureSize：Sobel算子的大小
          L2gradient ]]])   # 参数(布尔值)：
                              true： 使用更精确的L2范数进行计算（即两个方向的倒数的平方和再开放），
                              false：使用L1范数（直接将两个方向导数的绝对值相加）。
"""

import cv2
import numpy as np  
 
original_img = cv2.imread("test.jpg", 0)

# canny(): 边缘检测
img1 = cv2.GaussianBlur(original_img,(7,7),0) # (3,3) 表示高斯矩阵的长与宽, 0 是标准差
cv2.imshow('a',img1)
canny = cv2.Canny(img1, 50, 150)

# 形态学：边缘检测
# _,Thr_img = cv2.threshold(original_img,210,255,cv2.THRESH_BINARY)#设定红色通道阈值210（阈值影响梯度运算效果）
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))         #定义矩形结构元素
# gradient = cv2.morphologyEx(Thr_img, cv2.MORPH_GRADIENT, kernel) #梯度

# cv2.imshow("original_img", original_img) 
# cv2.imshow("gradient", gradient) 
cv2.imshow('Canny', canny)

cv2.waitKey(0)
cv2.destroyAllWindows()



