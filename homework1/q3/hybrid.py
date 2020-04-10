import cv2
import numpy as np

def cross_correlation_2d(img, kernel):
    # 互相关
    img_array = np.array(img)  # 把图像转换为数字
    r= img_array.shape[0]
    c = img_array.shape[1]  # 图像的列
    h = img_array.shape[2]  # 图像的高度
    r2 = kernel.shape[0]  # 核的行
    c2 = kernel.shape[1]  # 核的列
    new1 = np.zeros((r, (int)(c2 / 2)), np.int)  #获得一个新的空白矩阵
    new2= np.zeros(((int)(r2/ 2), c + new1.shape[1] * 2), np.int)
    conv = np.zeros((r, c, h))
    for i in range(3):#对矩阵进行一个互相关运算
        temp_img_array = np.hstack([new1, np.hstack([img_array[:, :, i], new1])]) #补零
        new_img_array = np.vstack([new2, np.vstack([temp_img_array, new2])])
        for j in range(r):
            for k in range(c):
                conv[j][k][i] = min(max(0,(new_img_array[j:j + r2, k:k + c2]* kernel).sum()),255)
    
    return conv
 
def convolve_2d(img, kernel):# 卷积
    kernel2 = np.rot90(np.fliplr(kernel), 2) #将图片进行2次逆时针90度翻转

    return cross_correlation_2d(img, kernel2)  
 
def gaussian_blur(img, sigma, height, width): 
    # 产生一个高斯核
    gaussian_kernel = np.zeros((height, width), dtype='double')
    center_row = height/2
    center_column = width/2
    s = 2*(sigma**2)
    for i in range(height):
        for j in range(width):
            x = i - center_row
            y = j - center_column
            gaussian_kernel[i][j] = (1.0/(np.pi*s))*np.exp(-float(x**2+y**2)/s)

    return convolve_2d(img,gaussian_kernel)   # 返回滤波后的图片

def scale(img, scale=2):
    # 把img2的尺寸调整为与img1一样，并且都缩小一倍
    img1 = cv2.imread(img[0])/255
    x, y = img1.shape[0:2]
    img1 = cv2.resize(img1, (int(y/scale), int(x/scale)))
    img2 = cv2.imread(img[1])/255
    img2 = cv2.resize(img2, (int(y/scale), int(x/scale)))

    return img1, img2

def hybrid(img1, img2):
    # 混合两张图片
    # 当low的sigma变大时，low变模糊，需要到更远处才能看到low的
    low = gaussian_blur(img1,9,73,73)
    cv2.imshow('low-pass', low)

    # 当high的sigma变大时，high变清楚，需要到更远处才能看到low的
    high = img2 - gaussian_blur(img2, 5,41,41)
    cv2.imshow('high-pass', high)

    return low + high

def visualization(image):
    # 可视化显示了5幅缩小的图像，以模拟从远处观看图像。
    num = 5  # Number of images to display.
    gap = 2  # Gap between images (px).

    # Create list of images.
    image_list = [image]
    max_height = image.shape[0]
    max_width = image.shape[1]

    # Add images to list and increase max width.
    for i in range(1, num):
        tmp = cv2.resize(image, (0, 0), fx=0.5 ** i, fy=0.5 ** i)
        max_width += tmp.shape[1] + gap
        image_list.append(tmp)
        # Create space for image stack.
    stack = np.ones((max_height, max_width, 3)) * 255

    # Add images to stack.
    current_x = 0
    for img in image_list:
        stack[
            max_height - img.shape[0] :, current_x : img.shape[1] + current_x, :
        ] = img
        current_x += img.shape[1] + gap

    return stack


img1, img2 = scale(['tiger.png', 'cat.png'])

result = hybrid(img1, img2)
cv2.imwrite('hybrid.jpg', result*255)
cv2.imshow('hybrid', result)

show = visualization(result)
cv2.imwrite('show.jpg', show*255)
cv2.imshow('visualize', show)

cv2.waitKey(0)
cv2.destroyAllWindows()
print('8888')

