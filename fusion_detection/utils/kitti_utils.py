# utils/kitti_utils.py
import numpy as np
import os
import cv2

class KittiCalibration:
    def __init__(self, calib_file):
        with open(calib_file, 'r') as f:
            lines = f.readlines()
        
        self.P0 = np.array([float(x) for x in lines[0].strip().split(' ')[1:13]]).reshape(3, 4)
        self.P1 = np.array([float(x) for x in lines[1].strip().split(' ')[1:13]]).reshape(3, 4)
        self.P2 = np.array([float(x) for x in lines[2].strip().split(' ')[1:13]]).reshape(3, 4)
        self.P3 = np.array([float(x) for x in lines[3].strip().split(' ')[1:13]]).reshape(3, 4)
        self.R0_rect = np.array([float(x) for x in lines[4].strip().split(' ')[1:10]]).reshape(3, 3)
        self.Tr_velo_to_cam = np.array([float(x) for x in lines[5].strip().split(' ')[1:13]]).reshape(3, 4)
        
        # Create 4x4 transformation matrix from velodyne to camera coordinates
        self.Tr_velo_to_cam_4x4 = np.eye(4)
        self.Tr_velo_to_cam_4x4[:3, :4] = self.Tr_velo_to_cam
        
        # Create 4x4 rectification matrix
        self.R0_rect_4x4 = np.eye(4)
        self.R0_rect_4x4[:3, :3] = self.R0_rect
    
    def project_velo_to_image(self, points_3d):
        """Project 3D points from velodyne coordinate to camera image space"""
        # Convert to homogeneous coordinates
        points_3d_homog = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
        
        # Velodyne -> Camera coordinate
        points_camera = np.dot(points_3d_homog, self.Tr_velo_to_cam_4x4.T)
        
        # Apply rectification
        points_rect = np.dot(points_camera, self.R0_rect_4x4.T)
        
        # Project to image plane
        points_2d = np.dot(points_rect[:, :3], self.P2[:3, :3].T) + self.P2[:3, 3]
        
        # Normalize
        depth = points_2d[:, 2]
        points_2d = points_2d[:, :2] / depth[:, np.newaxis]
        
        return points_2d, depth

def read_velodyne_points(velodyne_file):
    """Read velodyne point cloud from binary file"""
    points = np.fromfile(velodyne_file, dtype=np.float32).reshape(-1, 4)
    return points

def read_label(label_file):
    """Read KITTI label file"""
    labels = []
    with open(label_file, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split(' ')
            cls = parts[0]
            # Skip DontCare
            if cls == 'DontCare':
                continue
            
            bbox = np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])
            
            # Format: class, bbox (x1,y1,x2,y2), truncation, occlusion, alpha, dimensions, location, rotation_y
            labels.append({
                'type': cls,
                'bbox': bbox,  # [x1, y1, x2, y2]
                'truncation': float(parts[1]),
                'occlusion': int(parts[2]),
                'alpha': float(parts[3]),
                'dimensions': [float(parts[8]), float(parts[9]), float(parts[10])],  # [h, w, l]
                'location': [float(parts[11]), float(parts[12]), float(parts[13])],  # [x, y, z]
                'rotation_y': float(parts[14])
            })
    return labels
