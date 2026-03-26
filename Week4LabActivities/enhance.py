# enhance.py
from PIL import Image, ImageEnhance

# import an image
image = Image.open('Assets/cat.jpg')

# create an enhancer
vibrance_enhancer = ImageEnhance.Color(image)
contrast_enhancer = ImageEnhance.Contrast(image)
brightness_enhancer = ImageEnhance.Brightness(image)
sharpness_enhancer = ImageEnhance.Sharpness(image)

# apply the enhancer
enhanced_image = sharpness_enhancer.enhance(1.5)

# show
image.show()
enhanced_image.show()
