# utils.py
import torchvision.transforms.functional as TF
from PIL import Image

def center_crop_square_cfp(img: Image.Image):
    w, h = img.size 
    crop_size = h 
    return TF.center_crop(img, (crop_size, crop_size))

def center_crop_square_oct(img: Image.Image):
    w, h = img.size 
    crop_size = w  
    return TF.center_crop(img, (crop_size, crop_size))
