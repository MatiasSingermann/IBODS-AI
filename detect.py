import os
from helper_functions import drawBoundingBox

cwd = os.getcwd()
os.mkdir(f'{cwd}/result')

img = drawBoundingBox('IBODS_LBL-1/test/60_jpg.rf.af5b4d9b99351329bad5223344640506.jpg','logs/model1.tflite', 0.3, labels)
img.save(f'{cwd}/result/output.png')