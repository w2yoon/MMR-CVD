# test.py
import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from dataset import CFPOCTDataset
from utils import center_crop_square_cfp, center_crop_square_oct
from model import CFPOCTFusionNet

def get_transforms(train=False):
    if not train:
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
    else:
        cfp_transform, oct_transform = None, None
    return cfp_transform, oct_transform

def evaluate(model, test_loader, disease_names):
    model.eval()
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            cfp_inputs = batch['cfp'].to(device)
            oct_inputs = batch['oct'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(cfp_inputs, oct_inputs)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()
    
    all_preds = np.vstack(all_preds)
    all_probs = np.vstack(all_probs)
    
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print(f"F1 Score: {f1_score(all_labels, all_preds, average='micro'):.4f}")
    
    # Per-class Metrics: accuracy and F1-score
    num_classes = all_labels.shape[1]
    print("\nPer-class Metrics:")
    for i in range(num_classes):
        class_acc = accuracy_score(all_labels[:, i], all_preds[:, i])
        try:
            class_auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
        except ValueError:
            class_auc = float('nan')
        class_f1 = f1_score(all_labels[:, i], all_preds[:, i])
        print(f"Class {i} ({disease_names[i]}) -> Accuracy: {class_acc:.4f}, F1-score: {class_f1:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate CFP OCT Fusion Network")
    parser.add_argument('--csv_path', type=str, required=True, help="CSV file path with label information")
    parser.add_argument('--cfp_dir', type=str, required=True, help="CFP image directory")
    parser.add_argument('--oct_dir', type=str, required=True, help="OCT image directory")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the saved model checkpoint")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of workers for DataLoader")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.csv_path)
    cfp_transform, oct_transform = get_transforms(train=False)
    test_dataset = CFPOCTDataset(df, args.cfp_dir, args.oct_dir,
                                    cfp_transform=cfp_transform,
                                    oct_transform=oct_transform)
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = CFPOCTFusionNet(num_classes=len(test_dataset.mlb.classes_), fusion_dim=512).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print("Model loaded. Start evaluation...")
    evaluate(model, test_loader, test_dataset.mlb.classes_, device)
