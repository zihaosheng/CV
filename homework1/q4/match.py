'''
Harris角点检测以及匹配
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage, spatial
import math
    
def harris_detect(img,ksize=3,k = 0.04,threshold = 0.1,WITH_NMS = True ):
    '''
    自己实现角点检测
    
    params:
        img：灰度图片
        ksize：Sobel算子窗口大小
        k = 0.04 响应函数k
        threshold = 0.01 设定阈值  
        WITH_NMS = True 是否非极大值抑制
        
    return：
        R：响应值
        corner：与源图像一样大小，角点处像素值设置为255
        orientationImage：与源图像一样大小，各个角点的方向
    '''
     
    # 使用Sobel计算像素点x,y方向的梯度
    h,w = img.shape[:2]
    # Sobel函数求完导数后会有负值，还有会大于255的值。而原图像是uint8，即8位无符号数，
    # 所以Sobel建立的图像位数不够，会有截断。因此要使用16位有符号的数据类型，即cv2.CV_16S。
    grad = np.zeros((h,w,2),dtype=np.float32)
    grad[:,:,0] = cv2.Sobel(img,cv2.CV_16S,1,0,ksize)
    grad[:,:,1] = cv2.Sobel(img,cv2.CV_16S,0,1,ksize)

    # 计算Ix^2,Iy^2,Ix*Iy
    n = np.zeros((h,w,3),dtype=np.float32)
    m = np.zeros((h,w,3),dtype=np.float32)
    # print(m.shape)
    n[:,:,0] = grad[:,:,0]**2
    n[:,:,1] = grad[:,:,1]**2
    n[:,:,2] = grad[:,:,0]*grad[:,:,1]
        
    # 利用高斯函数对Ix^2,Iy^2,Ix*Iy进行滤波
    m[:,:,0] = cv2.GaussianBlur(n[:,:,0],ksize=(ksize,ksize),sigmaX=2)
    m[:,:,1] = cv2.GaussianBlur(n[:,:,1],ksize=(ksize,ksize),sigmaX=2)
    m[:,:,2] = cv2.GaussianBlur(n[:,:,2],ksize=(ksize,ksize),sigmaX=2)
    m = [np.array([[m[i,j,0],m[i,j,2]],[m[i,j,2],m[i,j,1]]]) for i in range(h) for j in range(w)]
    
    # 计算局部特征结果矩阵M的特征值和响应函数R(i,j)=det(M)-k(trace(M))^2  0.04<=k<=0.06
    D,T = list(map(np.linalg.det,m)),list(map(np.trace,m))
    R = np.array([d-k*t**2 for d,t in zip(D,T)])/4228250625

    # 将计算出响应函数的值R进行非极大值抑制，滤除一些不是角点的点，同时要满足大于设定的阈值
    #获取最大的R值
    R_max = np.max(R) 
    R = R.reshape(h,w)
    corner = np.zeros_like(R,dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if WITH_NMS:
                #除了进行进行阈值检测 还对3x3邻域内非极大值进行抑制(导致角点很小，会看不清)
                if R[i,j] > R_max*threshold and R[i,j] == np.max(R[max(0,i-1):min(i+2,h-1),max(0,j-1):min(j+2,w-1)]):
                    corner[i,j] = 255
                    # pass
            else:
                #只进行阈值检测
                if R[i,j] > R_max*threshold :
                    corner[i,j] = 255

    # 计算每个pixel的角度
    amplitudeImage = np.sqrt(n[:,:,0]+n[:,:,1])
    orientationImage = np.arctan2(grad[:,:,0], grad[:,:,1]) * 180/np.pi
    number_orientation = np.zeros(orientationImage.shape, np.int)
    for i in range(orientationImage.shape[0]):
        for j in range(orientationImage.shape[1]):
            if orientationImage[i][j] >= 22.5 and orientationImage[i][j] < 67.5:
                number_orientation[i][j] = 1
            elif orientationImage[i][j] >= 67.5 and orientationImage[i][j] < 112.5:
                number_orientation[i][j] = 2
            elif orientationImage[i][j] >= 112.5 and orientationImage[i][j] < 157.5:
                number_orientation[i][j] = 3
            elif orientationImage[i][j] >= 157.5 or orientationImage[i][j] < -157.5:
                number_orientation[i][j] = 4
            elif orientationImage[i][j] < -112.5 and orientationImage[i][j] >= -157.5:
                number_orientation[i][j] = 5
            elif orientationImage[i][j] < -67.5 and orientationImage[i][j] >= -112.5:
                number_orientation[i][j] = 6
            elif orientationImage[i][j] < -22.5 and orientationImage[i][j] >= -67.5:
                number_orientation[i][j] = 7
            else:
                number_orientation[i][j] = 0

    # 开始统计加权直方图
    for i in range(1, orientationImage.shape[0] - 1):
        for j in range(1, orientationImage.shape[1] - 1):
            if corner[i][j] != 0:
                histogram = np.zeros(8)
                for k in range(-1, 2):
                    for q in range(-1, 2):
                        histogram[number_orientation[i + k][j + q]] += amplitudeImage[i + k][j + q]

                # 最后的目标
                orientationImage[i][j] = histogram.argmax() * 45
    return R, corner, orientationImage

def scale_detecor(img,ksizes,k = 0.04):
    '''
    估计角点的scale
    
    params:
        img:灰度图片
        ksizes：窗口大小
        k = 0.04 响应函数k
        
    return：
        Respondes_max：与源图像一样大小，各个角点的大小
    '''
    # 使用Sobel计算像素点x,y方向的梯度
    h,w = img.shape[:2]
    Respondes = []
    temp = np.zeros((h, w))
    for ksize in ksizes:
        print(ksize)

        # 计算LoG
        img = cv2.GaussianBlur(img, ksize=(ksize, ksize), sigmaX=0)
        img = cv2.Laplacian(img, cv2.CV_16S, ksize=ksize)

        # 计算Sobel算子
        grad = np.zeros((h, w, 2), dtype=np.float32)
        grad[:,:,0] = cv2.Sobel(img,cv2.CV_16S,1,0,ksize)
        grad[:,:,1] = cv2.Sobel(img,cv2.CV_16S,0,1,ksize)

        # 计算Ix^2,Iy^2,Ix*Iy 
        m = np.zeros((h,w,3),dtype=np.float32)
        m[:,:,0] = grad[:,:,0]**2
        m[:,:,1] = grad[:,:,1]**2
        m[:,:,2] = grad[:,:,0]*grad[:,:,1]
            
        # 计算局部特征结果矩阵M的特征值和响应函数R(i,j)=det(M)-k(trace(M))^2  0.04<=k<=0.06
        m = [np.array([[m[i,j,0],m[i,j,2]],[m[i,j,2],m[i,j,1]]]) for i in range(h) for j in range(w)]
        D,T = list(map(np.linalg.det,m)),list(map(np.trace,m))
        # print(len(D)) # 307200
        R = np.array([d-k*t**2 for d,t in zip(D,T)])
        R = R.reshape(h, w)
        Respondes.append(R)

    temp = np.array([_ for _ in Respondes])

    # 计算所有R在每个坐标的最大值的索引
    Respondes_max = np.argmax(temp, axis=0)

    return Respondes_max

def detectKeypoints(image):
    '''
    Input:
        image -- BGR image with values between [0, 255]
    Output:
        list of detected keypoints, fill the cv2.KeyPoint objects with the
        coordinates of the detected keypoints, the angle of the gradient
        (in degrees), the detector response (Harris score for Harris detector)
        and set the size to 10.
    '''
    height, width = image.shape[:2]
    features = []
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    harrisImage, harrisMaxImage, orientationImage = harris_detect(grayImage)
    print('# of corners:', len(harrisMaxImage[harrisMaxImage>0]))
    ksizes = range(1, 20, 2)
    scale = scale_detecor(grayImage, ksizes)

    for y in range(height):
        for x in range(width):
            if harrisMaxImage[y, x] == 0:
                continue

            f = cv2.KeyPoint()

            # Fill in feature f with location and orientation
            # data here. Set f.size to 10, f.pt to the (x,y) coordinate,
            # f.angle to the orientation in degrees and f.response to
            # the Harris score
            f.size = (scale[y, x])*2 + 1
            f.pt = (x, y)
            f.angle = orientationImage[y, x]
            f.response = harrisImage[y, x]

            features.append(f)

    return features

def describeFeatures(image, keypoints):
    '''
    params:
        image： BGR图[0, 255]
        keypoints：检测到的特征，我们必须在指定的坐标计算特征描述符
    return:
        desc：K x W^2 数列，K是corner的个数
                W是窗口大小
    '''
    image = image.astype(np.float32)
    image /= 255.
    windowSize = 8
    desc = np.zeros((len(keypoints), windowSize * windowSize))
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayImage = ndimage.gaussian_filter(grayImage, 0.5)
    cnt = 0
    for i, f in enumerate(keypoints):
        x0, y0 = f.pt
        theta = - f.angle / 180.0 * np.pi
        T1 = np.array([[1, 0, -x0],[0, 1, -y0],[0, 0, 1]])
        cc = math.cos(theta)
        ss = math.sin(theta)
        R = np.array([[cc, -ss, 0], [ss, cc, 0], [0, 0, 1]])
        S = np.array([[0.2, 0, 0], [0, 0.2, 0], [0, 0, 1]])
        T2 = np.array([[1, 0, 4], [0, 1, 4], [0, 0, 1]])
        MF = np.dot(np.dot(np.dot(T2, S), R), T1)
        transMx = MF[0:2,0:3]
        destImage = cv2.warpAffine(grayImage, transMx,
            (windowSize, windowSize), flags=cv2.INTER_LINEAR)
        target = destImage[:8, :8]
        target = target - np.mean(target)
        if np.std(target) <= 10**(-5):
            desc[i, :] = np.zeros((windowSize * windowSize,))
        else:
            target = target / np.std(target)
            desc[i,:] = target.reshape(windowSize * windowSize)

    return desc

def matchFeatures(desc1, desc2):
    '''
    params:
        desc1：img1的descriptor
        desc2：img1的descriptor
    return:
        features matches: v2.DMatch的list
    '''
    matches = []
    # feature count = n
    assert desc1.ndim == 2
    # feature count = m
    assert desc2.ndim == 2
    # the two features should have the type
    assert desc1.shape[1] == desc2.shape[1]

    if desc1.shape[0] == 0 or desc2.shape[0] == 0:
        return []

    for i in range(desc1.shape[0]):
        u = desc1[i]
        diff = desc2 - u
        diff = diff ** 2
        sum_diff = diff.sum(axis = 1)
        dis = sum_diff ** 0.5
        j = np.argmin(dis)
        match = cv2.DMatch()
        match.queryIdx = i
        match.trainIdx = int(j)
        match.distance = dis[j]
        matches.append(match)

    return matches


if __name__=='__main__':
    img2 = cv2.imread('einstein.jpg')
    img1 = cv2.imread('img45.jpg')

    # 转换为灰度图像
    kp1 = detectKeypoints(img1)
    kp2 = detectKeypoints(img2)
    
    # descriptor
    des1 = describeFeatures(img1, kp1)
    des2 = describeFeatures(img2, kp2)

    # 画出角点
    cv2.drawKeypoints(image=img1, outImage=img1, keypoints=kp1,
                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                     color=(0, 0, 255))
    cv2.drawKeypoints(image=img2, outImage=img2, keypoints=kp2,
                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                         color=(0, 0, 255))
    # match
    matches = matchFeatures(des1, des2)
    img3=cv2.drawMatches(img1,kp1,img2,kp2,matches[:50],None,flags=2)
    plt.imshow(img3),plt.show()

    cv2.imwrite('match_rotation.jpg',img3)

