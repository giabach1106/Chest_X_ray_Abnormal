import cv2
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from tqdm import tqdm
import os
import warnings
import numpy as np
warnings.filterwarnings("ignore")

def read_xray(path, voi_lut=True, fix_monochrome=True, use_8bit=True, rescale_time=None):
    dicom = pydicom.read_file(path)
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
    
    data = data.astype(np.float64)
    if rescale_time:
        data = cv2.resize(data, (data.shape[1] // rescale_time, data.shape[0] // rescale_time), interpolation=cv2.INTER_CUBIC)

    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    data = data - np.min(data)
    data = data / np.max(data)
    if use_8bit:
        data = (data * 255).astype(np.uint8)
    else:
        data = (data * 65535).astype(np.uint16)
    return data



data_path = "/workspace/nabang1010/STEAM/LeGiaBach_STEAM/DATA/dataset/test"
data_image = "/workspace/nabang1010/STEAM/LeGiaBach_STEAM/DATA/dataset/test_png"

for i in tqdm(os.listdir(data_path)):
    # print(i)
    path = os.path.join(data_path, i)
    img_8bit = read_xray(path)
    cv2.imwrite(os.path.join(data_image , i[ :-5] + 'png'), img_8bit)
    # break
