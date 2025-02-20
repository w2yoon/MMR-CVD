# train.py
import argparse
import os
import pandas as pd
import torch
import torch.optim as optim
import torchvision.transforms as T

from dataset import FundusOCTDataset
from utils import center_crop_square_fundus, center_crop_square_oct
from model import FundusOCTFusionNet
from train import train  # train 함수는 별도의 파일로 분리한 경우 (여기서는 동일 train.py 내에 정의되어 있다면 그대로 사용)

def get_transforms(train=True):
    if train:
        fundus_transform = T.Compose([
            T.ToPILImage(),
            T.Lambda(center_crop_square_fundus),
            T.Resize((256, 256)),
            T.CenterCrop((224, 224)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([
                T.RandomRotation(degrees=15),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            ], p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        oct_transform = T.Compose([
            T.Lambda(center_crop_square_oct),
            T.Resize((256, 256)),
            T.CenterCrop((224, 224)),
            T.RandomApply([
                T.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                T.ColorJitter(brightness=0.1, contrast=0.1)
            ], p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
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
    return fundus_transform, oct_transform

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train FundusOCT Fusion Network")
    parser.add_argument('--csv_path', type=str, required=True, help="CSV file path with label information")
    parser.add_argument('--fundus_dir', type=str, required=True, help="Fundus image directory")
    parser.add_argument('--oct_dir', type=str, required=True, help="OCT image directory")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of workers for DataLoader")
    parser.add_argument('--save_path', type=str, default="model_checkpoint.pth", help="Path to save trained model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터셋 준비
    df = pd.read_csv(args.csv_path)
    fundus_transform, oct_transform = get_transforms(train=True)
    train_dataset = FundusOCTDataset(df, args.fundus_dir, args.oct_dir,
                                     fundus_transform=fundus_transform,
                                     oct_transform=oct_transform)
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # 모델, 옵티마이저, Loss 함수 초기화
    model = FundusOCTFusionNet(num_classes=len(train_dataset.mlb.classes_), fusion_dim=512).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.BCEWithLogitsLoss()

    # 학습 수행
    print("Start training...")
    train(model, train_loader, optimizer, criterion, args.epochs, device)
    torch.save(model.state_dict(), args.save_path)
    print(f"Training finished. Model saved to {args.save_path}")
