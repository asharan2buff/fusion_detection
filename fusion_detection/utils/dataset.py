# utils/dataset.py
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from .kitti_utils import KittiCalibration, read_velodyne_points, read_label
from .augmentation import apply_augmentations

class KittiMultiModalDataset(Dataset):
    def __init__(self, lidar_dir, calibration_dir, left_image_dir, labels_dir, 
                 num_points=20000, image_size=(416, 416), classes=None, 
                 shuffle=False, apply_augmentation=False, robustness_augmentations=None):
        self.lidar_dir = lidar_dir
        self.calibration_dir = calibration_dir
        self.left_image_dir = left_image_dir
        self.labels_dir = labels_dir
        
        self.num_points = num_points
        self.image_size = image_size
        self.apply_augmentation = apply_augmentation
        self.robustness_augmentations = robustness_augmentations or []
        
        if classes is None:
            self.classes = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
        else:
            self.classes = classes
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Get all file IDs
        self.sample_ids = [os.path.splitext(f)[0] for f in os.listdir(left_image_dir) if f.endswith('.png')]
        
        if shuffle:
            np.random.shuffle(self.sample_ids)
            
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        
        # Load calibration
        calib_path = os.path.join(self.calibration_dir, f"{sample_id}.txt")
        calib = KittiCalibration(calib_path)
        
        # Load image
        img_path = os.path.join(self.left_image_dir, f"{sample_id}.png")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Original image dimensions
        orig_h, orig_w = img.shape[:2]
        
        # Load point cloud
        lidar_path = os.path.join(self.lidar_dir, f"{sample_id}.bin")
        points = read_velodyne_points(lidar_path)
        
        # Load labels
        label_path = os.path.join(self.labels_dir, f"{sample_id}.txt")
        labels = read_label(label_path)
        
        # Project point cloud to image
        pc_points = points[:, :3]  # x, y, z
        img_points, depths = calib.project_velo_to_image(pc_points)
        
        # Filter points that are in front of the camera and within image bounds
        valid_inds = (img_points[:, 0] >= 0) & (img_points[:, 0] < orig_w) & \
                     (img_points[:, 1] >= 0) & (img_points[:, 1] < orig_h) & \
                     (depths > 0)
        
        valid_points = pc_points[valid_inds]
        valid_img_points = img_points[valid_inds]
        valid_depths = depths[valid_inds]
        
        # If too few points, pad with zeros
        if len(valid_points) < self.num_points:
            # Repeat points if not enough
            indices = np.random.choice(len(valid_points), self.num_points, replace=True)
        else:
            # Randomly sample points
            indices = np.random.choice(len(valid_points), self.num_points, replace=False)
        
        sampled_points = valid_points[indices]
        sampled_img_points = valid_img_points[indices]
        sampled_depths = valid_depths[indices]
        
        # Prepare point cloud features (x, y, z, r where r is reflectance)
        point_features = np.zeros((self.num_points, 4))
        point_features[:, :3] = sampled_points
        # Add reflectance if available
        if points.shape[1] > 3:
            point_features[:, 3] = points[valid_inds][indices, 3]
        
        # Scale image points to match YOLO grid
        scaled_img_points = sampled_img_points.copy()
        scaled_img_points[:, 0] *= self.image_size[0] / orig_w
        scaled_img_points[:, 1] *= self.image_size[1] / orig_h
        
        # Resize image
        img_resized = cv2.resize(img, self.image_size)
        
        # Apply augmentations if enabled
        if self.apply_augmentation:
            img_resized = apply_augmentations(img_resized, self.robustness_augmentations)
        
        # Normalize image
        img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)) / 255.0
        
        # Prepare target boxes for YOLO
        target_boxes = []
        for label in labels:
            if label['type'] in self.class_to_idx:
                cls_id = self.class_to_idx[label['type']]
                # Convert KITTI format [x1, y1, x2, y2] to YOLO format [x_center, y_center, width, height]
                x1, y1, x2, y2 = label['bbox']
                
                # Scale to [0, 1]
                x1, x2 = x1 / orig_w, x2 / orig_w
                y1, y2 = y1 / orig_h, y2 / orig_h
                
                # Convert to center format
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                
                target_boxes.append([cls_id, x_center, y_center, width, height])
        
        target_boxes = torch.tensor(target_boxes, dtype=torch.float32)
        
        return {
            'image': img_tensor,
            'point_cloud': torch.from_numpy(point_features).float(),
            'img_points': torch.from_numpy(scaled_img_points).float(),
            'depths': torch.from_numpy(sampled_depths).float(),
            'boxes': target_boxes,
            'image_id': sample_id,
            'orig_size': (orig_h, orig_w)
        }
