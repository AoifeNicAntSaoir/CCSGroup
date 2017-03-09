from resizeimage import resizeimage
from PIL import Image

with open('testImage.png', 'r+b') as f:
    with Image.open(f) as image:
        cover = resizeimage.resize_cover(image, [8, 8])

        cover.save('test-image.png', image.format)

