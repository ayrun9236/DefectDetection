import cv2
import numpy as np
import random
import os
from pathlib import Path

def create_random_defect(height, width, min_area=2500):
    """
    Create a random defect mask with area larger than min_area
    Returns a binary mask with the defect
    """
    # Create an empty mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Generate random parameters for the defect
    center_x = random.randint(0, width-1)
    center_y = random.randint(0, height-1)
    
    # Generate random shape (either ellipse or polygon)
    shape_type = random.choice(['ellipse', 'polygon'])
    
    if shape_type == 'ellipse':
        # Keep generating until we get a defect with sufficient area
        while True:
            axes_length = (
                random.randint(30, 100),  # major axis
                random.randint(30, 100)   # minor axis
            )
            angle = random.randint(0, 360)
            
            # Draw the ellipse
            cv2.ellipse(mask, (center_x, center_y), axes_length, 
                       angle, 0, 360, 255, -1)
            
            # Check area
            if cv2.countNonZero(mask) >= min_area:
                break
            mask.fill(0)  # Clear and try again
            
    else:  # polygon
        while True:
            # Generate 3-6 points for the polygon
            num_points = random.randint(3, 6)
            points = []
            for _ in range(num_points):
                point_x = center_x + random.randint(-100, 100)
                point_y = center_y + random.randint(-100, 100)
                points.append([point_x, point_y])
            
            # Draw the polygon
            points = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)
            
            # Check area
            if cv2.countNonZero(mask) >= min_area:
                break
            mask.fill(0)  # Clear and try again
    
    return mask

def apply_defect(image, mask):
    """Apply the defect mask to the image"""
    # Create a darker region for the defect
    darkening_factor = random.uniform(0.3, 0.7)
    defect_image = image.copy()
    defect_image[mask > 0] = defect_image[mask > 0] * darkening_factor
    return defect_image

def generate_defect_images(master_image_path, output_dir, num_images=2000):
    """Generate multiple defect images from a master image"""
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read the master image
    master_img = cv2.imread(master_image_path)
    if master_img is None:
        raise ValueError(f"Could not read master image: {master_image_path}")
    
    height, width = master_img.shape[:2]
    
    # Generate images
    for i in range(num_images):
        # Decide number of defects (0-5)
        num_defects = random.randint(0, 5)
        
        # Create a copy of master image
        result_img = master_img.copy()
        
        # Generate and apply defects
        for _ in range(num_defects):
            defect_mask = create_random_defect(height, width)
            result_img = apply_defect(result_img, defect_mask)
        
        # Save the image
        output_path = output_dir / f"defect_{i:04d}_n{num_defects}.jpg"
        cv2.imwrite(str(output_path), result_img)
        
        # Print progress every 100 images
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_images} images")

if __name__ == "__main__":
    master_path = "master.jpg"
    output_dir = "generated_defects"
    
    try:
        generate_defect_images(master_path, output_dir)
        print("Defect image generation completed successfully!")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
