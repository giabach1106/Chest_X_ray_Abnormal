import cv2
import darknet
import os
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
from tqdm import tqdm


cfg_file_path = "/workspace/nabang1010/STEAM/LeGiaBach_STEAM/src/yolov4/darknet/chestxray.cfg"
data_file_path = "workspace/nabang1010/STEAM/LeGiaBach_STEAM/src/yolov4/darknet/chestxray.data"
weight_file_path = "/workspace/nabang1010/STEAM/LeGiaBach_STEAM/src/yolov4/darknet/backup/chestxray_10000.weights"
network,  class_names, class_colors = darknet.load_network(cfg_file_path, data_file_path, weight_file_path, batch_size=1)

darknet_width = darknet.network_width(network)
darknet_height = darknet.network_height(network)

DF_RAW = pd.read_csv("/workspace/nabang1010/STEAM/LeGiaBach_STEAM/src/EfficientNetB4/submission.csv")


pbar = tqdm(range(len(DF_RAW)))
for i in pbar:
    img_file = DF_RAW.iloc[i, 0] + ".png"
    img_path = os.path.join("/workspace/nabang1010/STEAM/LeGiaBach_STEAM/DATA/dataset/test_png", img_file)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (darknet_width, darknet_height),
                                interpolation=cv2.INTER_LINEAR)
    
    img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
    darknet.copy_image_from_bytes(img_for_detect, img_resized.tobytes())
    detections = darknet.detect_image(network, class_names, img_for_detect, thresh=0.25)
    darknet.free_image(img_for_detect)
    pbar.set_description("Processing {}".format(img_file))
