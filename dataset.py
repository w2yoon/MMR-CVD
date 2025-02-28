# dataset.py
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer

def contrast_enhance(image):
    scale = 300
    blur_img = cv2.GaussianBlur(image, (0, 0), scale / 30)
    merge_img = cv2.addWeighted(image, 4, blur_img, -4, 128)
    return merge_img

class CFPOCTDataset(Dataset):
    def __init__(self, df, cfp_image_dir, oct_image_dir, mlb=None, cfp_transform=None, oct_transform=None):
        self.df = df
        self.cfp_image_dir = cfp_image_dir
        self.oct_image_dir = oct_image_dir
        self.cfp_transform = cfp_transform
        self.oct_transform = oct_transform
        self.mlb = mlb if mlb is not None else MultiLabelBinarizer()
        if not mlb:
            self.label = self.mlb.fit_transform(df['Disease'].str.split(','))
        else:
            self.label = self.mlb.transform(df['Disease'].str.split(','))

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # ----- CFP Image ----- #
        cfp_img_name = self.df.iloc[idx, 2]
        cfp_image_path = os.path.join(self.cfp_image_dir, cfp_img_name[:2], cfp_img_name)
        cfp_image = Image.open(cfp_image_path)
        cfp_image = np.array(cfp_image)
        cfp_image = contrast_enhance(cfp_image)
        cfp_image = cv2.cvtColor(cfp_image, cv2.COLOR_BGR2RGB)
        if self.cfp_transform:
            cfp_image = self.cfp_transform(cfp_image)
        
        # ----- OCT Image ----- #
        zip_name = self.df.iloc[idx, 3]
        oct_img_name = zip_name.split('.')[0]
        oct_image_folder = os.path.join(self.oct_image_dir, oct_img_name[:2], oct_img_name)
        oct_files = sorted(os.listdir(oct_image_folder))
        oct_64_files = oct_files[::3]
        
        oct_3d = []
        for filename in tqdm(oct_64_files, desc=f"Loading OCT slices for {oct_img_name}", leave=False):
            image_path = os.path.join(oct_image_folder, filename)
            image = Image.open(image_path).convert('L')
            if self.oct_transform:
                image = self.oct_transform(image)
            oct_3d.append(image)
        
        oct_3d = torch.stack(oct_3d, dim=0)  
        oct_3d = oct_3d.permute(1, 0, 2, 3).contiguous() 
        
        label = torch.tensor(self.label[idx], dtype=torch.float32)
        
        sample = {
            'cfp': cfp_image,
            'oct': oct_3d,
            'label': label
        }
        return sample
