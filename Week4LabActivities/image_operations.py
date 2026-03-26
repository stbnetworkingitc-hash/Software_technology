#image_operations.py
from PIL import Image, ImageOps

# create an image
image = Image.open('Assets/dog.jpg')

# position changes
image_mirror = ImageOps.mirror(image)
# image_scale = ImageOps.scale(image, 0.5)

# color changes
image_inverted = ImageOps.invert(image_mirror)

# add and remove
image_border = ImageOps.expand(
    image = image_inverted,
    border = 50,
    fill = (255,255,255))
# image_padded = ImageOps.pad(image, (4000,6000))
# image_crop = ImageOps.crop(image = image, border = 200)
image_border.show()