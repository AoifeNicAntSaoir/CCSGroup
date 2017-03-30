from PIL import ImageChops, Image

img1 = Image.open("photo_1.jpg")
img2 = Image.open("photo_2.jpg")

print ImageChops.difference(img2,img1).getbbox()#returns none if identical
                                                #bbox - bounding box

