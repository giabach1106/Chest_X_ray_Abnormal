import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import os
from tqdm import tqdm


df = pd.read_csv('/workspace/nabang1010/STEAM/LeGiaBach_STEAM/DATA/dataset/train.csv')


df["width"] = np.nan
df["height"] = np.nan


for i in tqdm(range(len(df))):
    image_id = df.iloc[i,:]['image_id']
    img = Image.open('/workspace/nabang1010/STEAM/LeGiaBach_STEAM/DATA/dataset/images/' + image_id + '.png')
    img = ImageOps.exif_transpose(img)
    width, height = img.size
    df.iloc[i, df.columns.get_loc('width')] = width
    df.iloc[i, df.columns.get_loc('height')] = height


df.to_csv('/workspace/nabang1010/STEAM/LeGiaBach_STEAM/DATA/dataset/train_w_h.csv', index=False)