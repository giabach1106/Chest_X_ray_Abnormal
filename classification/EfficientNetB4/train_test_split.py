import os
import glob2
import numpy as np
import math
import pandas as pd
import shutil
from tqdm import tqdm

DF_RAW = pd.read_csv("/workspace/nabang1010/STEAM/LeGiaBach_STEAM/DATA/train_droped.csv")
ROOT_IMAGE_FD = "/workspace/nabang1010/STEAM/LeGiaBach_STEAM/DATA/dataset/images"
TRAIN_FD = "/workspace/nabang1010/STEAM/LeGiaBach_STEAM/DATA/data_classify/train"
VAL_FD = "/workspace/nabang1010/STEAM/LeGiaBach_STEAM/DATA/data_classify/val"


list_image_abnormal = DF_RAW[DF_RAW["label"] == 1]["image_id"].tolist()
list_image_noraml = DF_RAW[DF_RAW["label"] == 0]["image_id"].tolist()

nb_train_abnormal = math.floor(len(list_image_abnormal)*0.75)
rand_idx_train_abnormal = np.random.randint(0, len(list_image_abnormal), nb_train_abnormal)

nb_train_normal = math.floor(len(list_image_noraml)*0.75)
rand_idx_train_normal = np.random.randint(0, len(list_image_noraml), nb_train_normal)


for idx in tqdm(range(len(list_image_abnormal))):
    image_file = list_image_abnormal[idx] + ".png"
    if idx in rand_idx_train_abnormal:
        shutil.copy(os.path.join(ROOT_IMAGE_FD, image_file), os.path.join(TRAIN_FD, "1", image_file))
    else:
        shutil.copy(os.path.join(ROOT_IMAGE_FD, image_file), os.path.join(VAL_FD, "1", image_file))

for idx in tqdm(range(len(list_image_noraml))):
    image_file = list_image_noraml[idx] + ".png"
    if idx in rand_idx_train_normal:
        shutil.copy(os.path.join(ROOT_IMAGE_FD, image_file), os.path.join(TRAIN_FD, "0", image_file))
    else:
        shutil.copy(os.path.join(ROOT_IMAGE_FD, image_file), os.path.join(VAL_FD, "0", image_file))
