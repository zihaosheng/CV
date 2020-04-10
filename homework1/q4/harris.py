'''
Harris角点检测
'''
import cv2
import numpy as np

def harris_detect(img,ksize=3,k = 0.04,threshold = 0.1,WITH_NMS = True ):
    '''
    自己实现角点检测
    
    params:
        img:灰度图片
        ksize：Sobel算子窗口大小
        k = 0.04 响应函数k
        threshold = 0.01 设定阈值  
        WITH_NMS = True 是否非极大值抑制
        
    return：
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
    # print(len(D)) # 307200
    R = np.array([d-k*t**2 for d,t in zip(D,T)])/4228250625

    # 将计算出响应函数的值R进行非极大值抑制，滤除一些不是角点的点，同时要满足大于设定的阈值
    # 获取最大的R值
    R_max = np.max(R)    
    R = R.reshape(h,w)
    corner = np.zeros_like(R,dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if WITH_NMS:
                #除了进行进行阈值检测 还对3x3邻域内非极大值进行抑制(导致角点很小，会看不清)
                if R[i,j] > R_max*threshold and R[i,j] == np.max(R[max(0,i-1):min(i+2,h-1),max(0,j-1):min(j+2,w-1)]):
                    corner[i,j] = 255
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
            # 分成8个角度
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
    return corner, orientationImage

def scale_detecor(img,ksizes,k = 0.04,):
    '''
    估计角点的scale
    
    params:
        img:灰度图片
        ksizes：窗口大小
        k = 0.04 响应函数k
        
    return：
        Respondes_max：与源图像一样大小，各个角点的大小
    '''
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


    
if __name__=='__main__':

    img = cv2.imread('einstein.jpg')
    print(img.shape,'1111')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # 转换为灰度图像
    dst, angle = harris_detect(gray,k=0.04,threshold=0.1)
    img[dst>0.01*dst.max()] = [255,255,255]
    print('the # of corners:', len(img[dst>0.01*dst.max()]))
    qw = 1
    x, y = angle.shape

    # 画出scale
    ksizes = range(3, 20, 2)
    radius = scale_detecor(gray, ksizes, )
    for i in range(x):
        for j in range(y):
            if dst[i][j]>0.01*dst.max():
                # 画出scale
                scale = (radius[i][j])*2 + 1
                print('scale:',qw,':',scale)
                qw += 1
                cv2.circle(img, (j, i), scale, (0, 0, 255))

                # 画出每个corner的方向
                theta = angle[i][j] * np.pi / 180
                print(theta)
                cv2.line(img, (j, i), (np.max([np.min([j+int(scale*np.cos(theta)),y-1]),0]), np.max([np.min([i+int(scale*np.sin(theta)),x-1]),0])),(0,0,255))

    cv2.imwrite('einstein_harris.jpg',img)
    cv2.imshow('',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()