"""
pip install opencv-contrib-python
"""


import cv2
from cv2 import dnn_superres
import os
import glob
import matplotlib.pyplot as plt
sr = dnn_superres.DnnSuperResImpl_create()


MainPath = os.path.normpath(os.getcwd()) + os.sep
Images = 'PicsNamed' + os.sep 

Result = 'Results' + os.sep 

Files = glob.glob(MainPath  + Images + '*.png', recursive=True)


for file in Files:
    
    file_name = os.path.split(file)[1]
    
    image_file = file
    # image_file = "./upscaled.png"
    
    image = cv2.imread(image_file )
    
    path = "EDSR_x4.pb"
    sr.readModel(path)
    sr.setModel("edsr", 3)
    # sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    result = sr.upsample(image)
    
    RGB_result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    plt.imshow(RGB_result)
    
    # Save the image
    cv2.imwrite(MainPath  + Result + file_name, result)


# plt.imshow(image)

