import glob as gb    #导入glob模块
import cv2
import os


# 返回该路径下所有的 jpg 文件的路径
img_path = gb.glob("./data/test_dataset/*.jpg")
for path in img_path:
    (filepath, tempfilename) = os.path.split(path)
    (filename, extension) = os.path.splitext(tempfilename)
