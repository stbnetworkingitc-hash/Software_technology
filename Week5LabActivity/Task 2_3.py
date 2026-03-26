#your answer
#task2_3.py
import cv2
image = cv2.imread('Assets/Insects/1_Gammar 2021_1_2021_06_03-13-19-23-879.png')
resized_img = cv2.resize(image, (400, 300))
cv2.imshow('Resized Image', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()