import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from data_loader import generate_synthetic_dataset, AlphaFoldDataset
from model import AlphaFoldCNN

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description="Train AlphaFold-CNN for Protein Thermostability Prediction")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--size", type=int, default=128, help="Dimension of input matrices (Size x Size)")
    parser.add_argument("--use-synthetic", action="store_true", default=True, help="Force use of synthetic mock data (default: True)")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of synthetic samples to generate")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Training split ratio")
    parser.add_argument("--model-path", type=str, default="best_model.pth", help="Path to save the best model weights")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing protein data")
    return parser.parse_args()


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculates Accuracy, Precision, Recall, and F1-Score for binary predictions.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, 
                    optimizer: optim.Optimizer, device: torch.device) -> Tuple[float, Dict[str, float]]:
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []
    
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1) # shape: (batch_size, 1)
        
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        # Calculate predicted classes
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        
        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        
    epoch_loss = running_loss / len(loader.dataset)
    metrics = calculate_metrics(np.array(all_targets), np.array(all_preds))
    
    return epoch_loss, metrics


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module, 
             device: torch.device) -> Tuple[float, Dict[str, float]]:
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []
    
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1)
        
        logits = model(inputs)
        loss = criterion(logits, targets)
        
        running_loss += loss.item() * inputs.size(0)
        
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        
        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        
    val_loss = running_loss / len(loader.dataset)
    metrics = calculate_metrics(np.array(all_targets), np.array(all_preds))
    
    return val_loss, metrics


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Using execution device: {device}")
    
    # 1. Obtain data (Synthetic Fallback)
    data_list = []
    if args.use_synthetic:
        synthetic_dir = os.path.join(args.data_dir, "synthetic")
        data_list = generate_synthetic_dataset(synthetic_dir, num_samples=args.num_samples)
    else:
        # User could configure real data paths, we check if they exist
        # If not, fall back to synthetic
        real_data_exists = False
        if os.path.exists(args.data_dir):
            files = [f for f in os.listdir(args.data_dir) if f.endswith(".pdb")]
            if len(files) > 0:
                real_data_exists = True
                print(f"[+] Found {len(files)} local PDB structures in {args.data_dir}. Loading them...")
                # Try to pair with PAE files
                for f in files:
                    pdb_path = os.path.join(args.data_dir, f)
                    pae_path = pdb_path.replace(".pdb", "_pae.json").replace("-model.pdb", "-predicted_aligned_error.json")
                    if not os.path.exists(pae_path):
                        pae_path = None
                    # Label: for real data, assume a naming pattern or default to 0/1 randomly for demo
                    # For example, if filename contains "thermo", label 1, else 0
                    label = 1 if "thermo" in f.lower() else (1 if np.random.rand() > 0.5 else 0)
                    data_list.append({
                        "pdb_path": pdb_path,
                        "pae_path": pae_path,
                        "label": label
                    })
        
        if not real_data_exists:
            print("[-] No local data found and --use-synthetic not disabled. Falling back to synthetic simulation...")
            synthetic_dir = os.path.join(args.data_dir, "synthetic")
            data_list = generate_synthetic_dataset(synthetic_dir, num_samples=args.num_samples)
            
    # 2. Build datasets & loaders
    full_dataset = AlphaFoldDataset(data_list, size=args.size)
    train_size = int(args.split_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"[+] Dataset configured: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")
    
    # 3. Model, Optimizer, Loss
    model = AlphaFoldCNN(size=args.size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    
    # 4. Training Loop
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "train_f1": [], "val_f1": []
    }
    best_val_loss = float("inf")
    
    print("\n" + "="*80)
    print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Val Loss':<10} | {'Train Acc':<9} | {'Val Acc':<9} | {'Val F1':<8}")
    print("="*80)
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_f1"].append(val_metrics["f1"])
        
        print(f"{epoch:<6} | {train_loss:<10.4f} | {val_loss:<10.4f} | {train_metrics['accuracy']:<9.2%} | {val_metrics['accuracy']:<9.2%} | {val_metrics['f1']:<8.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.model_path)
            # print(f"    [Checkpoint] Saved new best model to {args.model_path}")
            
    print("="*80)
    print(f"[+] Training completed. Best validation loss: {best_val_loss:.4f}")
    print(f"[+] Model weights saved to {args.model_path}")
    
    # 5. Plot training results
    epochs_range = range(1, args.epochs + 1)
    plt.figure(figsize=(12, 5))
    
    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss", color="royalblue")
    plt.plot(epochs_range, history["val_loss"], label="Val Loss", color="crimson")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Progression")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["train_acc"], label="Train Accuracy", color="royalblue")
    plt.plot(epochs_range, history["val_acc"], label="Val Accuracy", color="crimson")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Progression")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    curves_path = os.path.join(args.data_dir, "training_curves.png")
    os.makedirs(os.path.dirname(curves_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(curves_path)
    print(f"[+] Training curves saved to {curves_path}")


if __name__ == "__main__":
    main()
