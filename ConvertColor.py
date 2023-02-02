from PIL import Image
import PIL.ImageOps as IO

## This python code for invert the image color 
x = 'axx.png'
#img_name = x
#img = Image.open(img_name)
#if img.mode == 'RGBA':
#    r,g,b,a = img.split()
#    rgb_img = Image.merge('RGB', (r,g,b))
#    img = IO.invert(rgb_img)
#    img.save(img_name)

# #convert back
img_name = x
img = Image.open(img_name)
if img.mode == 'RGB':
    img = IO.invert(img)
    r,g,b = img.split()
    a_channel = Image.new('L', img.size, 255)
    rgba_img = Image.merge('RGBA', (r,g,b,a_channel))
    rgba_img.save(img_name)
