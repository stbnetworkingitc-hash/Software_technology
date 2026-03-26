#your answer
#task2_6.py
import cv2
image = cv2.imread('Assets/Insects/1_Gammar 2021_1_2021_06_03-13-19-23-879.png')
print(f"Original pixel value at (50, 50): {image[50, 50]}")
image[50, 50] = [255, 255, 255]
cv2.imwrite('modified.png', image)