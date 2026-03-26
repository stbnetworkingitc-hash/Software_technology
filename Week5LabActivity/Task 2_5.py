#your answer
#task2_5.py
import cv2
image = cv2.imread('Assets/Insects/1_Gammar 2021_1_2021_06_03-13-19-23-879.png')
image_copy = image.copy()
cv2.circle(image_copy, (100, 100), 50, (0, 0,
255), 3) # BGR: Red
cv2.imshow('Image with Circle', image_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()