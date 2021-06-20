import os
import re
import numpy as np



path = "D:/CycleGAN_model/images_Z/frame30/10000"
paths = [os.path.join(path , i ) for i in os.listdir(path) if re.search(".png$", i )]
## 정렬 작업
store1 = []
store2 = []
idx_list = []
"""
for file in paths:
    #print(file)
    idx = int(file[len(path+"fake_B_"):-4])
    idx_list.append(idx)

print(list(np.sort(idx_list)))
idx_list = list(np.sort(idx_list))
result_list =[]

for file in idx_list:
    result_list.append(path+"fake_B_"+str(file)+".png")
"""

import pandas as pd
df = pd.DataFrame(paths)
df.to_csv("test.csv")
print(len(paths))
#len('ims/2/a/2a.2710.png')

pathIn= "D:/CycleGAN_model/images_Z/frame30/10000/"
pathOut = "D:/CycleGAN_model/images_Z/frame30/atest_10000.mp4"
fps =5
import cv2
frame_array = []
for idx , path in enumerate(sorted(paths)) :
    if (idx % 2 == 0) | (idx % 5 == 0) :
        continue
    img = cv2.imread(path)

    height, width, layers = img.shape
    size = (width,height)
    frame_array.append(img)
out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
out.release()