import xml.etree.ElementTree as ET
from os import getcwd
import os
import numpy as np

wd = getcwd()
classes = ["mask"]

def convert_annotation(image_id, list_file):
    in_file = open('/Users/zihaosheng/Documents/研究生课程/CV/final_proj/MaskDetect-YOLOv4-PyTorch-master/images/label_4/Annotations/%s.xml'%(image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()
    list_file.write('datasets/train/JPEGImages/%s.jpg'%( image_id))
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

    list_file.write('\n')


path = '/Users/zihaosheng/Documents/研究生课程/CV/final_proj/MaskDetect-YOLOv4-PyTorch-master/images/label_4/JPEGImages'
img_paths = os.listdir(path)

image_ids = np.sort(list(map(lambda x: x.split('.')[0], img_paths)))
list_file = open('train_black4.txt', 'w')
for image_id in image_ids:
    if image_id=='':
        continue
    else:
        print(image_id)
        convert_annotation(image_id, list_file)
list_file.close()
# print(len(image_ids))
# print(image_ids)