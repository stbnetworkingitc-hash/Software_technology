#image_filter.py
from PIL import Image, ImageFilter

# create an image
image = Image.open('Assets/cat.jpg')

# apply a basic filter
#image_blur = image.filter(ImageFilter.BLUR)
#image_contour = image.filter(ImageFilter.CONTOUR)x
#image_emboss = image.filter(ImageFilter.EMBOSS)
#image_edge = image.filter(ImageFilter.FIND_EDGES)

# apply advanced filters
image_boxblur = image.filter(ImageFilter.BoxBlur(radius = 20))
image_gaussianblur = image.filter(ImageFilter.GaussianBlur(radius= 20))
image_unsharp = image.filter(ImageFilter.UnsharpMask(radius = 20))

image_boxblur.show()
image_gaussianblur.show()