from PIL import ImageChops, Image

img1 = Image.open("images/1.png")
img2 = Image.open("images/2.png")

print ImageChops.difference(img2,img1).getbbox()#returns none if identical
                                                #bbox - bounding box

