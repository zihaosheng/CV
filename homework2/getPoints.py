import cv2
img = cv2.imread('uttower1.jpg')
a =[]
b = []
def on_EVENT_LBUTTONDOWN(event, x, y,flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        a.append(x)
        b.append(y)
        cv2.circle(img, (x, y), 5, (255, 255, 255), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    3.0, (255, 255, 255), thickness=3)
        cv2.imwrite('points1.png', img)
        cv2.imshow("image1", img)

cv2.namedWindow("image1")
cv2.setMouseCallback("image1", on_EVENT_LBUTTONDOWN)
cv2.imshow("image1", img)
cv2.waitKey(0)
print (a,b)

img2 = cv2.imread('uttower2.jpg')
a2 =[]
b2 = []
def on_EVENT_LBUTTONDOWN(event, x, y,flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        a2.append(x)
        b2.append(y)
        cv2.circle(img2, (x, y), 5, (255, 255, 255), thickness=-1)
        cv2.putText(img2, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    3.0, (255, 255, 255), thickness=3)
        cv2.imwrite('points1.png', img)
        cv2.imshow("image2", img2)

cv2.namedWindow("image2")
cv2.setMouseCallback("image2", on_EVENT_LBUTTONDOWN)
cv2.imshow("image2", img2)
cv2.waitKey(0)
cv2.imwrite('points2.png', img2)
print (a2,b2)

def get_correspondences():
    return a, b, a2, b2

