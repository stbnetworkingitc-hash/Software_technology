#intro.py
from PIL import Image

# create an image via import
image = Image.open('Assets/Insects/1_Gammar 2021_1_2021_06_03-13-19-23-879.PNG')

# analyze the image
print(image.size)
print(image.filename)
print(image.format)

#to flip the image
image = image.transpose(Image.Transpose.ROTATE_90)

#to show the image
image.show()

# exercise
#this will rotate the image 30 degree and save the output image.
img_rotated = image.rotate(30)
img_rotated.save('img_rotated.png', 'png')

