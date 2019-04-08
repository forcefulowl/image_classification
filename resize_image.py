from PIL import Image
import os, sys

path = 'C:\\Users\gavin\Desktop\copy_224\\NV\\'
target_path = 'C:\\Users\gavin\Desktop\whole_256\\NV\\'
dirs = os.listdir(path)

def resize():
    i = 0
    for item in dirs:
        if item.endswith('.jpg'):
            im = Image.open(path+item)
            imResize = im.resize((256, 256), Image.ANTIALIAS)
            imResize.save(target_path+item, 'JPEG', quality=90)
            i = i + 1

resize()

