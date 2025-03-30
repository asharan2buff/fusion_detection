# Create a script called split_dataset.py
import os
import shutil
import numpy as np

def create_train_val_split(data_dir, val_ratio=0.2, seed=42):
    # Get all file IDs
    image_dir = os.path.join(data_dir, 'left_images/training/image_2')
    all_files = [f.split('.')[0] for f in os.listdir(image_dir) if f.endswith('.png')]
    
    # Shuffle files
    np.random.seed(seed)
    np.random.shuffle(all_files)
    
    # Split into train and validation
    split_idx = int(len(all_files) * (1 - val_ratio))
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    # Create validation directories
    val_dirs = [
        'calibration/validation/calib',
        'left_images/validation/image_2',
        'labels/validation/label_2',
        'velodyne/validation/velodyne'
    ]
    
    for dir_path in val_dirs:
        os.makedirs(os.path.join(data_dir, dir_path), exist_ok=True)
    
    # Move validation files
    for file_id in val_files:
        # Move calibration
        shutil.move(
            os.path.join(data_dir, f'calibration/training/calib/{file_id}.txt'),
            os.path.join(data_dir, f'calibration/validation/calib/{file_id}.txt')
        )
        
        # Move image
        shutil.move(
            os.path.join(data_dir, f'left_images/training/image_2/{file_id}.png'),
            os.path.join(data_dir, f'left_images/validation/image_2/{file_id}.png')
        )
        
        # Move label
        shutil.move(
            os.path.join(data_dir, f'labels/training/label_2/{file_id}.txt'),
            os.path.join(data_dir, f'labels/validation/label_2/{file_id}.txt')
        )
        
        # Move velodyne
        shutil.move(
            os.path.join(data_dir, f'velodyne/training/velodyne/{file_id}.bin'),
            os.path.join(data_dir, f'velodyne/validation/velodyne/{file_id}.bin')
        )
    
    print(f"Split complete: {len(train_files)} training samples, {len(val_files)} validation samples")

if __name__ == "__main__":
    create_train_val_split('data')
