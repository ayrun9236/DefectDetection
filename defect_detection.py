import cv2
import numpy as np
from pathlib import Path
import json

class DefectDetector:
    def __init__(self, master_path):
        """Initialize the detector with a master image"""
        self.master_img = cv2.imread(master_path)
        if self.master_img is None:
            raise ValueError(f"Could not read master image: {master_path}")
        
        # Convert master image to grayscale
        self.master_gray = cv2.cvtColor(self.master_img, cv2.COLOR_BGR2GRAY)
    
    def detect_defects(self, defect_image_path, threshold=30):
        """
        Detect defects by comparing with master image
        Args:
            defect_image_path: Path to the defect image
            threshold: Threshold for difference detection (default: 30)
        Returns:
            contours: List of detected contours
            diff_mask: Binary mask showing differences
            annotated_image: Original image with contours drawn
        """
        # Read defect image
        defect_img = cv2.imread(str(defect_image_path))
        if defect_img is None:
            raise ValueError(f"Could not read defect image: {defect_image_path}")
        
        # Convert to grayscale
        defect_gray = cv2.cvtColor(defect_img, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(self.master_gray, defect_gray)
        
        # Apply threshold to get binary mask
        _, diff_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Apply some morphological operations to clean up the mask
        kernel = np.ones((3,3), np.uint8)
        diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel)
        diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on the original image
        annotated_image = defect_img.copy()
        cv2.drawContours(annotated_image, contours, -1, (0, 255, 0), 2)
        
        return contours, diff_mask, annotated_image
    
    def analyze_defects(self, contours):
        """
        Analyze detected defects and return their properties
        """
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

def process_defect_images(master_path, defects_dir, output_dir, threshold=30):
    """
    Process all defect images in a directory and save results
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector
    detector = DefectDetector(master_path)
    
    # Process all images in defects directory
    defects_dir = Path(defects_dir)
    results = {}
    
    for img_path in defects_dir.glob("*.jpg"):
        try:
            # Detect defects
            contours, diff_mask, annotated_image = detector.detect_defects(img_path, threshold)
            
            # Analyze defects
            defect_properties = detector.analyze_defects(contours)
            
            # Save results
            base_name = img_path.stem
            
            # Save annotated image
            cv2.imwrite(str(output_dir / f"{base_name}_annotated.jpg"), annotated_image)
            
            # Save difference mask
            cv2.imwrite(str(output_dir / f"{base_name}_mask.jpg"), diff_mask)
            
            # Store analysis results
            results[base_name] = {
                "num_defects": len(contours),
                "defects": defect_properties
            }
            
            print(f"Processed {img_path.name}")
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")
    
    # Save analysis results to JSON
    with open(output_dir / "defect_analysis.json", 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    master_path = "master.jpg"
    defects_dir = "generated_defects"
    output_dir = "detection_results"
    
    try:
        process_defect_images(master_path, defects_dir, output_dir)
        print("Defect detection completed successfully!")
    except Exception as e:
        print(f"Error occurred: {str(e)}") 