'''
DoG in SIFT角点检测以及匹配
'''
import cv2
import numpy as np

# 读取图片
img2 = cv2.imread("einstein.jpg")
img1 = cv2.imread("img_sca.jpg")

# 转为灰度图像
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 创建一个sift对象 并计算灰度图像
sift = cv2.xfeatures2d_SIFT.create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# 匹配
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
goodMatch = []
for m, n in matches:
    if m.distance < 0.50*n.distance:
        goodMatch.append(m)

# 增加一个维度
goodMatch = np.expand_dims(goodMatch, 1)

img_out = cv2.drawMatchesKnn(img1, kp1, img2, kp2, goodMatch[:150], None, flags=2)
cv2.imwrite('Dog-match_scale.jpg', img_out)
cv2.imshow('image', img_out)#展示图片



# 在图像上绘制关键点
# DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS表示对每个关键点画出圆圈和方向
img = cv2.drawKeypoints(image=img2, outImage=img2, keypoints=kp2,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                        color=(51, 163, 236))
# cv2.imwrite("sift_keypoints.jpg", img)
cv2.imshow("sift_keypoints", img)
cv2.waitKey()
cv2.destroyAllWindows()



