import cv2

img=cv2.imread("galaxy.jpg",0)
 

resized_img=cv2.resize(img,(500,500))
cv2.imshow("Galaxy",resized_img)
img.imwrite("galaxy_resized.jpg",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
