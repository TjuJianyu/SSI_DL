import cv2
import os
from tqdm import tqdm
try:
    os.mkdir("../out/resize_lips/")
    os.mkdir("../out/resize_tongue/")
except:
    pass

for i in tqdm(range(1,68146+1)): #[9598,9599]:
    try:
        name = str(i)
        for i in range(6-len(name)):
            name = "0"+name
        impath = "../data/Songs_Lips/%s.tif" % name
        lips = cv2.imread(impath)
        lips = lips[20:360,100:500,:]
        lips = cv2.resize(lips,(50,42),interpolation=cv2.INTER_AREA)
        cv2.imwrite("../out/resize_lips/%s.tif" % name, lips)


        impath = "../data/Songs_Tongue/%s.tif" % name
        tongue = cv2.imread(impath)
        tongue = tongue[30:200,50:250,:]
        tongue = cv2.resize(tongue,(50,42),interpolation=cv2.INTER_AREA)
        cv2.imwrite("../out/resize_tongue/%s.tif" % name,tongue)
    except:
        print(i,name)
        