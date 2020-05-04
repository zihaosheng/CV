import cv2
import numpy as np

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

def get_H(a, b, a2, b2):
    moving_points = np.array([[a[0],b[0]],[a[1],b[1]], [a[2],b[2]], [a[3],b[3]]])
    fixed_poins = np.array([[a2[0],b2[0]],[a2[1],b2[1]], [a2[2],b2[2]], [a2[3],b2[3]]])
    H = get_perspective_mat(moving_points, fixed_poins)
    return H


def draw_correspondences(img, img2, H, a, b):
    colors = [(0,0,255), (255,0,0), (0,255,0), (0,0,0)]
    for i in range(len(a)):
        cv2.circle(img, (a[i], b[i]), radius=5, color=colors[i])
        aaa = np.mat(H) * np.mat([a[i], b[i], 1]).T
        cv2.circle(img2, (int(aaa[0, 0] / aaa[-1, 0]), int(aaa[1, 0] / aaa[-1, 0])), radius=5, color=colors[i])
    img_compare = np.hstack([img, img2])
    cv2.imshow('compare', img_compare)
    cv2.waitKey(0)

def imageBoundingBox(img, H):

    corner1 = np.array((0, 0, 1))
    corner2 = np.array((0, img.shape[0] - 1, 1))
    corner3 = np.array((img.shape[1] - 1, img.shape[0] - 1, 1))
    corner4 = np.array((img.shape[1] - 1, 0, 1))

    p1 = np.dot(H, corner1)
    p2 = np.dot(H, corner2)
    p3 = np.dot(H, corner3)
    p4 = np.dot(H, corner4)

    p1 /= p1[2]
    p2 /= p2[2]
    p3 /= p3[2]
    p4 /= p4[2]

    minY = min(p1[0], p2[0], p3[0], p4[0])
    minX = min(p1[1], p2[1], p3[1], p4[1])
    maxY = max(p1[0], p2[0], p3[0], p4[0])
    maxX = max(p1[1], p2[1], p3[1], p4[1])

    return int(minX), int(minY), int(maxX), int(maxY)

def warp_and_blend(img, img2, H):
    minX, minY, maxX, maxY = imageBoundingBox(img, H)
    print(minX, minY, maxX, maxY)
    x, y, z = img.shape
    new_img = np.zeros((np.maximum(x, maxX)-np.minimum(0, minX)+1, np.maximum(y, maxY)-np.minimum(0, minY)+1, z))
    new_img[np.maximum(0, -minX):np.maximum(0, -minX)+x, :y,:] = img2
    for i in range(x):
        for j in range(y):
            trans = np.mat(H) * np.mat([j, i, 1]).T
            right_horizon = int(trans[0, 0] / trans[-1, 0])
            right_vertica = int(trans[1, 0] / trans[-1, 0])
            if right_vertica - 1-np.minimum(0, minX) > 0 and right_vertica + 1-np.minimum(0, minX) < new_img.shape[0]\
                    and right_horizon+1 < new_img.shape[1]:
                new_img[right_vertica-np.minimum(0, minX), right_horizon, :] = img[i, j, :]
                new_img[right_vertica + 1-np.minimum(0, minX), right_horizon, :] = img[i, j, :]
                new_img[right_vertica - 1-np.minimum(0, minX), right_horizon, :] = img[i, j, :]
                new_img[right_vertica-np.minimum(0, minX), right_horizon + 1, :] = img[i, j, :]
                new_img[right_vertica-np.minimum(0, minX), right_horizon - 1, :] = img[i, j, :]
                new_img[right_vertica + 1-np.minimum(0, minX), right_horizon + 1, :] = img[i, j, :]
                new_img[right_vertica - 1-np.minimum(0, minX), right_horizon - 1, :] = img[i, j, :]
                new_img[right_vertica - 1-np.minimum(0, minX), right_horizon + 1, :] = img[i, j, :]
                new_img[right_vertica - 1-np.minimum(0, minX), right_horizon - 1, :] = img[i, j, :]

    cv2.imwrite('another.png', new_img)


if __name__=='__main__':
    img = cv2.imread('room1.jpg')
    img2 = cv2.imread('room2.jpg')
    a = []
    b = []
    a2 = []
    b2 = []
    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            a.append(x)
            b.append(y)
            cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
            cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 0), thickness=1)
            cv2.imshow("image1", img)
    cv2.namedWindow("image1")
    cv2.setMouseCallback("image1", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image1", img)
    cv2.waitKey(0)
    print(a, b)
    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            a2.append(x)
            b2.append(y)
            cv2.circle(img2, (x, y), 1, (255, 0, 0), thickness=-1)
            cv2.putText(img2, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 0), thickness=1)
            cv2.imshow("image2", img2)
    cv2.namedWindow("image2")
    cv2.setMouseCallback("image2", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image2", img2)
    cv2.waitKey(0)
    print(a2, b2)
    H = get_H(a, b, a2, b2)
    draw_correspondences(img, img2, H, a, b)
    warp_and_blend(img, img2, H)