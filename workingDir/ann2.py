from PIL import Image

img = Image.open('testImage.png').convert('L')
newImg = img.resize((8,8))
img.show();
