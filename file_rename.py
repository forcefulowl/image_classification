import os

path = "C:\\Users\gavin\Desktop\Test_set_2"

files = os.listdir(path)

i = 1

for file in files:
    os.rename(os.path.join(path, file), os.path.join(path, str(i)+'.jpg'))
    i = i+1

