import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import time

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        
        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        
        # Decoder
        self.dec1 = DoubleConv(512 + 256, 256)
        self.dec2 = DoubleConv(256 + 128, 128)
        self.dec3 = DoubleConv(128 + 64, 64)
        
        # Final layer
        self.final = nn.Conv2d(64, 1, kernel_size=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Decoder with dynamic resizing
        # Upsample e4 and concatenate with e3
        d1 = F.interpolate(e4, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d1 = self.dec1(torch.cat([d1, e3], dim=1))
        
        # Upsample d1 and concatenate with e2
        d2 = F.interpolate(d1, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        # Upsample d2 and concatenate with e1
        d3 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d3 = self.dec3(torch.cat([d3, e1], dim=1))
        
        # Final layer
        out = torch.sigmoid(self.final(d3))
        return out

class DefectDataset(Dataset):
    def __init__(self, master_path, defect_paths, mask_paths, transform=None, target_size=(512, 512)):
        self.master_path = master_path
        self.defect_paths = defect_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.target_size = target_size
        
        # Load master image once
        self.master_img = cv2.imread(str(master_path))
        if self.master_img is None:
            raise ValueError(f"Could not read master image: {master_path}")
        self.master_img = cv2.cvtColor(self.master_img, cv2.COLOR_BGR2RGB)
        self.master_img = cv2.resize(self.master_img, target_size)
        
    def __len__(self):
        return len(self.defect_paths)
    
    def __getitem__(self, idx):
        # Load defect image
        defect_img = cv2.imread(str(self.defect_paths[idx]))
        if defect_img is None:
            raise ValueError(f"Could not read defect image: {self.defect_paths[idx]}")
        defect_img = cv2.cvtColor(defect_img, cv2.COLOR_BGR2RGB)
        defect_img = cv2.resize(defect_img, self.target_size)
        
        # Load mask
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask image: {self.mask_paths[idx]}")
        mask = cv2.resize(mask, self.target_size)
        mask = mask / 255.0  # Normalize to [0, 1]
        
        if self.transform:
            defect_img = self.transform(defect_img)
            self.master_img_t = self.transform(self.master_img)
            mask = torch.from_numpy(mask).float()
        
        # Stack master and defect images along channel dimension
        x = torch.cat([self.master_img_t, defect_img], dim=0)
        return x, mask.unsqueeze(0)

def train_model(model, train_loader, val_loader, device, num_epochs=3, target_accuracy=98.0):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Calculate total steps for progress tracking
    total_train_steps = len(train_loader)
    total_val_steps = len(val_loader)
    
    print("\nStarting training...")
    print(f"Total training batches per epoch: {total_train_steps}")
    print(f"Total validation batches per epoch: {total_val_steps}")
    print(f"Training on device: {device}")
    print(f"Target validation accuracy: {target_accuracy}%")
    print(f"Models will be saved in: {models_dir}/")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        correct_pixels = 0
        total_pixels = 0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("Training phase:")
        
        # Training progress bar
        train_pbar = tqdm(train_loader, total=total_train_steps, 
                         desc=f"Training Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (inputs, masks) in enumerate(train_pbar):
            inputs, masks = inputs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            train_loss += loss.item()
            pred = (outputs > 0.5).float()
            correct_pixels += (pred == masks).sum().item()
            total_pixels += masks.numel()
            
            # Update progress bar
            current_loss = train_loss / (batch_idx + 1)
            current_acc = 100. * correct_pixels / total_pixels
            train_pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })
        
        train_pbar.close()
        avg_train_loss = train_loss / total_train_steps
        train_accuracy = 100. * correct_pixels / total_pixels
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct_pixels = 0
        total_pixels = 0
        
        print("Validation phase:")
        val_pbar = tqdm(val_loader, total=total_val_steps,
                       desc=f"Validation Epoch {epoch+1}/{num_epochs}")
        
        with torch.no_grad():
            for batch_idx, (inputs, masks) in enumerate(val_pbar):
                inputs, masks = inputs.to(device), masks.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, masks)
                
                # Calculate metrics
                val_loss += loss.item()
                pred = (outputs > 0.5).float()
                correct_pixels += (pred == masks).sum().item()
                total_pixels += masks.numel()
                
                # Update progress bar
                current_loss = val_loss / (batch_idx + 1)
                current_acc = 100. * correct_pixels / total_pixels
                val_pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{current_acc:.2f}%'
                })
        
        val_pbar.close()
        avg_val_loss = val_loss / total_val_steps
        val_accuracy = 100. * correct_pixels / total_pixels
        val_losses.append(avg_val_loss)
        
        # Save model for this epoch
        epoch_model_path = models_dir / f"model_epoch_{epoch+1:03d}_acc_{val_accuracy:.2f}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'total_time': time.time() - start_time
        }, epoch_model_path)
        print(f"✓ Saved epoch model to {epoch_model_path}")
        
        # Save best model if this is the best validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = models_dir / "best_model.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'total_time': time.time() - start_time
            }, best_model_path)
            print("✓ Saved new best model!")
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time
        
        # Print epoch summary
        print("\nEpoch Summary:")
        print(f"Training Loss: {avg_train_loss:.4f} | Training Accuracy: {train_accuracy:.2f}%")
        print(f"Validation Loss: {avg_val_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%")
        print(f"Epoch Time: {epoch_time:.1f}s | Total Time: {total_time:.1f}s")
        print("-" * 80)
        
        # Check if validation accuracy exceeds target
        if val_accuracy >= target_accuracy:
            print(f"\n🎯 Target validation accuracy of {target_accuracy}% reached!")
            print(f"Training stopped at epoch {epoch+1}")
            print(f"Final validation accuracy: {val_accuracy:.2f}%")
            print(f"Total training time: {total_time:.1f}s")
            break
    
    # Save final model
    final_model_path = models_dir / "final_model.pth"
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
        'total_time': time.time() - start_time
    }, final_model_path)
    print(f"\n✓ Saved final model to {final_model_path}")
    
    # Print final training summary
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Total training time: {total_time:.1f}s")
    print(f"Models saved in: {models_dir}/")
    
    return train_losses, val_losses

def prepare_data(master_path, defects_dir, detection_results_dir):
    defects_dir = Path(defects_dir)
    detection_results_dir = Path(detection_results_dir)
    
    # Get all defect images and their corresponding masks
    defect_paths = list(defects_dir.glob('*.jpg'))
    mask_paths = [detection_results_dir / f"{p.stem}_mask.jpg" for p in defect_paths]
    
    # Filter out pairs where both files exist
    valid_pairs = [(d, m) for d, m in zip(defect_paths, mask_paths) if d.exists() and m.exists()]
    if not valid_pairs:
        raise ValueError("No valid image-mask pairs found!")
    
    defect_paths, mask_paths = zip(*valid_pairs)
    print(f"Found {len(defect_paths)} valid image-mask pairs")
    
    # Split into train and validation sets
    train_defects, val_defects, train_masks, val_masks = train_test_split(
        defect_paths, mask_paths, test_size=0.2, random_state=42
    )
    
    return train_defects, val_defects, train_masks, val_masks

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    master_path = "master.jpg"
    defects_dir = "generated_defects"
    detection_results_dir = "detection_results"
    model_save_path = "defect_model.pth"
    
    try:
        # Prepare data
        train_defects, val_defects, train_masks, val_masks = prepare_data(
            master_path, defects_dir, detection_results_dir
        )
        
        # Define transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets with fixed size
        target_size = (512, 512)  # Use power of 2 for better feature map alignment
        train_dataset = DefectDataset(master_path, train_defects, train_masks, transform, target_size)
        val_dataset = DefectDataset(master_path, val_defects, val_masks, transform, target_size)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
        
        # Initialize model
        model = UNet(in_channels=6).to(device)
        
        # Train model with target accuracy
        train_losses, val_losses = train_model(
            model, 
            train_loader, 
            val_loader, 
            device,
            target_accuracy=98.0  # Set target accuracy to 98%
        )
        
        # Save final model
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
        
        # Plot training history
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.savefig('training_history.png')
        plt.close()
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 