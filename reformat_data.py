import os
import shutil


f = open('C:\\Users\gavin\Desktop\labels.txt')
line = f.readline()
labels = []

while line:
    if line[13] == '1':
        labels.append(0)
    elif line[15] == '1':
        labels.append(1)
    elif line[17] == '1':
        labels.append(2)
    elif line[19] == '1':
        labels.append(3)
    elif line[21] == '1':
        labels.append(4)
    elif line[23] == '1':
        labels.append(5)
    elif line[25] == '1':
        labels.append(6)
    line = f.readline()

path = 'C:\\Users\gavin\Desktop\ISIC2018_Task3_Training_Input'
count = 0
while count < len(labels):
    curr_num = str(24306+count)
    if labels[count] == 0:
        shutil.copy(os.path.join(path, 'ISIC_00'+curr_num+'.jpg'), 'C:\\Users\gavin\Desktop\Train\MEL')
    if labels[count] == 1:
        shutil.copy(os.path.join(path, 'ISIC_00'+curr_num+'.jpg'), r'C:\\Users\gavin\Desktop\Train\NV')
    if labels[count] == 2:
        shutil.copy(os.path.join(path, 'ISIC_00'+curr_num+'.jpg'), 'C:\\Users\gavin\Desktop\Train\BCC')
    if labels[count] == 3:
        shutil.copy(os.path.join(path, 'ISIC_00'+curr_num+'.jpg'), 'C:\\Users\gavin\Desktop\Train\AKIEC')
    if labels[count] == 4:
        shutil.copy(os.path.join(path, 'ISIC_00'+curr_num+'.jpg'), 'C:\\Users\gavin\Desktop\Train\BKL')
    if labels[count] == 5:
        shutil.copy(os.path.join(path, 'ISIC_00'+curr_num+'.jpg'), 'C:\\Users\gavin\Desktop\Train\DF')
    if labels[count] == 6:
        shutil.copy(os.path.join(path, 'ISIC_00'+curr_num+'.jpg'), 'C:\\Users\gavin\Desktop\Train\VASC')
    count = count + 1