# test.py
import argparse
import os
import pandas as pd
import torch
import torchvision.transforms as T

from dataset import FundusOCTDataset
from utils import center_crop_square_fundus, center_crop_square_oct
from model import FundusOCTFusionNet
from test import evaluate  # 평가 함수가 별도의 파일로 분리되어 있다면 import

def get_transforms(train=False):
    if not train:
        fundus_transform = T.Compose([
            T.ToPILImage(),
            T.Lambda(center_crop_square_fundus),
            T.Resize((256, 256)),
            T.CenterCrop((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        oct_transform = T.Compose([
            T.Lambda(center_crop_square_oct),
            T.Resize((256, 256)),
            T.CenterCrop((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        # (테스트에서는 기본적으로 train=False 사용)
        fundus_transform, oct_transform = None, None
    return fundus_transform, oct_transform

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate FundusOCT Fusion Network")
    parser.add_argument('--csv_path', type=str, required=True, help="CSV file path with label information")
    parser.add_argument('--fundus_dir', type=str, required=True, help="Fundus image directory")
    parser.add_argument('--oct_dir', type=str, required=True, help="OCT image directory")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the saved model checkpoint")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of workers for DataLoader")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터셋 준비 (테스트 전용 transform)
    df = pd.read_csv(args.csv_path)
    fundus_transform, oct_transform = get_transforms(train=False)
    test_dataset = FundusOCTDataset(df, args.fundus_dir, args.oct_dir,
                                    fundus_transform=fundus_transform,
                                    oct_transform=oct_transform)
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 모델 초기화 및 checkpoint 로드
    model = FundusOCTFusionNet(num_classes=len(test_dataset.mlb.classes_), fusion_dim=512).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print("Model loaded. Start evaluation...")
    evaluate(model, test_loader, test_dataset.mlb.classes_, device)
