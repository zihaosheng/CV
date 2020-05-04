import cv2
import numpy as np

def get_corners(img):
    corner1 = np.array((0, 0, 1))
    corner2 = np.array((0, img.shape[0] - 1, 1))
    corner3 = np.array((img.shape[1] - 1, img.shape[0] - 1, 1))
    corner4 = np.array((img.shape[1] - 1, 0, 1))
    return corner1, corner2, corner3, corner4

def get_perspective_mat(points0, points1):
    assert points0.shape == points1.shape and points0.shape[0]>=4
    nums = points0.shape[0]
    coefficient_mat = np.zeros((2*nums, 8))
    b = np.zeros((2*nums, 1))
    for i in range(nums):
        currentPoint = points0[i, :]
        currentPoint1 = points1[i, :]
        coefficient_mat[2*i, :] = [currentPoint[0], currentPoint[1], 1, 0,0,0,
                                   -currentPoint[0]*currentPoint1[0], -currentPoint[1]*currentPoint1[0]]
        b[2*i,0] = currentPoint1[0]
        coefficient_mat[2*i+1, :] = [0,0,0, currentPoint[0], currentPoint[1], 1,
                                     -currentPoint[0]*currentPoint1[1], -currentPoint[1]*currentPoint1[1]]
        b[2*i+1,0] = currentPoint1[1]

    perspective_mat = np.linalg.lstsq(coefficient_mat, b)[0]
    perspective_mat = np.insert(perspective_mat, 8, values=np.array([1]), axis=0)
    perspective_mat = perspective_mat.reshape((3,3))

    return perspective_mat

def get_H(moving_points, fixed_poins):
    H = get_perspective_mat(moving_points, fixed_poins)
    return H

if __name__=='__main__':
    img = cv2.imread('billboard.jpg')
    new_img = cv2.imread('billboard.jpg')
    a =[]
    b = []
    def on_EVENT_LBUTTONDOWN(event, x, y,flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            a.append(x)
            b.append(y)
            cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
            cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 0), thickness=1)
            cv2.imshow("image", img)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image", img)
    cv2.waitKey(0)

    img_star = cv2.imread('sjtu.jpg')
    corner1, corner2, corner3, corner4  = get_corners(img_star)
    moving_points = np.array([[corner1[0],corner1[1]],[corner2[0],corner2[1]],
                              [corner3[0],corner3[1]], [corner4[0],corner4[1]]])

    fixed_poins = np.array([[a[0],b[0]],[a[1],b[1]], [a[2],b[2]], [a[3],b[3]]])
    H = get_H(moving_points, fixed_poins)

    x, y, z = img_star.shape
    for i in range(x):
        for j in range(y):
            trans = np.mat(H)*np.mat([j, i , 1]).T
            right_horizon = int(trans[0, 0]/trans[-1, 0])
            right_vertica = int(trans[1, 0]/trans[-1, 0])
            new_img[right_vertica, right_horizon, :] = img_star[i, j, :]

    cv2.imwrite('sjtu_on_billboard.png', new_img)