import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from pathlib import Path
from defect_detection_dl import UNet
import json

class DefectPredictor:
    def __init__(self, model_path, master_path, device=None, target_size=(512, 512)):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.target_size = target_size
            
        # Load master image
        self.master_img = cv2.imread(str(master_path))
        if self.master_img is None:
            raise ValueError(f"Could not read master image: {master_path}")
        self.master_img = cv2.cvtColor(self.master_img, cv2.COLOR_BGR2RGB)
        self.original_size = self.master_img.shape[:2][::-1]  # (width, height)
        self.master_img = cv2.resize(self.master_img, target_size)
        
        # Load model
        self.model = UNet(in_channels=6).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            # New checkpoint format
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.epoch = checkpoint['epoch']
            self.metrics = {
                'train_loss': checkpoint['train_loss'],
                'val_loss': checkpoint['val_loss'],
                'train_accuracy': checkpoint['train_accuracy'],
                'val_accuracy': checkpoint['val_accuracy']
            }
            print(f"Loaded model from epoch {self.epoch} with validation accuracy: {self.metrics['val_accuracy']:.2f}%")
        else:
            # Old format (direct state dict)
            self.model.load_state_dict(checkpoint)
            print("Loaded model (old format)")
            
        self.model.eval()
        
        # Define transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Transform master image once
        self.master_tensor = self.transform(self.master_img).to(self.device)
    
    def predict(self, image_path, threshold=0.5):
        """
        Predict defects in the given image
        Returns:
            mask: Binary mask of predicted defects
            annotated_image: Original image with predicted defects outlined
            contours: List of detected contours
        """
        # Read and preprocess image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize for model
        img_resized = cv2.resize(original_img, self.target_size)
        img_tensor = self.transform(img_resized).to(self.device)
        
        # Stack master and input images
        x = torch.cat([self.master_tensor, img_tensor], dim=0).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            pred = self.model(x)
            pred = pred.squeeze().cpu().numpy()
        
        # Convert to binary mask
        mask = (pred > threshold).astype(np.uint8) * 255
        
        # Resize mask back to original size
        mask = cv2.resize(mask, self.original_size)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on original image
        annotated_image = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        cv2.drawContours(annotated_image, contours, -1, (0, 255, 0), 2)
        
        return mask, annotated_image, contours
    
    def analyze_defects(self, contours):
        """Analyze properties of detected defects"""
        defect_properties = []
        
        for i, contour in enumerate(contours):
            # Calculate basic properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate center point
            M = cv2.moments(contour)
            if M["m00"] != 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
            else:
                center_x = x + w//2
                center_y = y + h//2
            
            defect_properties.append({
                "id": i,
                "area": float(area),
                "perimeter": float(perimeter),
                "width": int(w),
                "height": int(h),
                "center": (int(center_x), int(center_y))
            })
        
        return defect_properties

def main():
    # Paths
    model_path = "models/best_model.pth"  # Updated path to use models directory
    master_path = "master.jpg"
    test_dir = "test_defects"  # Directory containing test images
    output_dir = "dl_detection_results"
    
    try:
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize predictor
        print("Initializing model...")
        target_size = (512, 512)  # Same as training
        predictor = DefectPredictor(model_path, master_path, target_size=target_size)
        print(f"Using device: {predictor.device}")
        
        # Process test images
        test_dir = Path(test_dir)
        if not test_dir.exists():
            raise ValueError(f"Test directory not found: {test_dir}")
        
        test_images = list(test_dir.glob("*.jpg"))
        if not test_images:
            raise ValueError(f"No jpg images found in {test_dir}")
        
        print(f"Found {len(test_images)} images to process")
        for img_path in test_images:
            try:
                print(f"Processing {img_path.name}...")
                # Predict defects
                mask, annotated_image, contours = predictor.predict(img_path)
                
                # Analyze defects
                defect_properties = predictor.analyze_defects(contours)
                
                # Save results
                base_name = img_path.stem
                cv2.imwrite(str(output_dir / f"{base_name}_pred_mask.jpg"), mask)
                cv2.imwrite(str(output_dir / f"{base_name}_pred_annotated.jpg"), annotated_image)
                
                # Save analysis results
                with open(output_dir / f"{base_name}_analysis.json", 'w') as f:
                    json.dump({
                        "num_defects": len(contours),
                        "defects": defect_properties
                    }, f, indent=4)
                
                print(f"Found {len(contours)} defects")
                
            except Exception as e:
                print(f"Error processing {img_path.name}: {str(e)}")
        
        print("Processing completed!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 