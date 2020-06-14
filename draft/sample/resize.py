from PIL import Image

img = Image.open('queries/banana.jpg') # image extension *.png,*.jpg
new_width  = 100
new_height = 100
img = img.resize((new_width, new_height), Image.ANTIALIAS)
img.save('queries/banana_100.jpg') 