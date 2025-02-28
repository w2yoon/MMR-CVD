# train.py
import argparse
import os
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

from dataset import CFPOCTDataset
from utils import center_crop_square_cfp, center_crop_square_oct
from model import CFPOCTFusionNet

def get_transforms(train=True):
    if train:
        cfp_transform = T.Compose([
            T.ToPILImage(),
            T.Lambda(center_crop_square_cfp),
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
        cfp_transform = T.Compose([
            T.ToPILImage(),
            T.Lambda(center_crop_square_cfp),
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
    return cfp_transform, oct_transform

def train(model, train_loader, optimizer, criterion, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            cfp_inputs = batch['cfp'].to(device)
            oct_inputs = batch['oct'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            scaler = torch.cuda.amp.GradScaler() 

            with torch.cuda.amp.autocast(): 
                outputs = model(cfp_inputs, oct_inputs)
                
            loss = criterion(outputs, labels)
            scaler.scale(loss).backward()  # Gradient Scaling
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train CFP OCT Fusion Network")
    parser.add_argument('--csv_path', type=str, required=True, help="CSV file path with label information")
    parser.add_argument('--cfp_dir', type=str, required=True, help="CFP image directory")
    parser.add_argument('--oct_dir', type=str, required=True, help="OCT image directory")
    parser.add_argument('--epochs', type=int, default=150, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of workers for DataLoader")
    parser.add_argument('--save_path', type=str, default="model_checkpoint.pth", help="Path to save trained model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.csv_path)
    cfp_transform, oct_transform = get_transforms(train=True)
    train_dataset = CFPOCTDataset(df, args.cfp_dir, args.oct_dir, cfp_transform=cfp_transform, oct_transform=oct_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = CFPOCTFusionNet(num_classes=len(train_dataset.mlb.classes_), fusion_dim=512).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.BCEWithLogitsLoss()

    print("Start training...")
    train(model, train_loader, optimizer, criterion, args.epochs, device)
    torch.save(model.state_dict(), args.save_path)
    print(f"Training finished. Model saved to {args.save_path}")
