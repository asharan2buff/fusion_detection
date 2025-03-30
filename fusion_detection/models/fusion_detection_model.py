# models/fusion_detection_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet_backbone import PointNetBackbone
from models.yolo_backbone import YOLOBackbone
from models.fusion_module import SpatialILFusionModule

class SpatialILFusionModel(nn.Module):
    def __init__(self, num_classes=8, num_points=20000, num_global_feats=1024, fusion_type='residual'):
        super(SpatialILFusionModel, self).__init__()
        
        # PointNet for LiDAR processing
        self.pointnet = PointNetBackbone(num_points=num_points, num_global_feats=num_global_feats)
        
        # YOLO for image processing
        self.yolo = YOLOBackbone(num_classes=num_classes)
        
        # Fusion module
        image_features_dims = {
            'large': 512,
            'medium': 256,
            'small': 128
        }
        
        self.fusion_module = SpatialILFusionModule(
            image_features_dims=image_features_dims,
            lidar_feat_dim=num_global_feats,
            fusion_type=fusion_type
        )
        
        # Final prediction layers (reusing YOLO prediction layers)
        self.pred_large = nn.Conv2d(512, 3 * (5 + num_classes), 1)
        self.pred_medium = nn.Conv2d(256, 3 * (5 + num_classes), 1)
        self.pred_small = nn.Conv2d(128, 3 * (5 + num_classes), 1)
        
    def forward(self, image, point_cloud, img_points):
        """
        Args:
            image: RGB image (B, 3, H, W)
            point_cloud: LiDAR point cloud (B, N, 4)
            img_points: Image coordinates for each LiDAR point (B, N, 2)
        """
        # Process point cloud with PointNet
        lidar_features, _ = self.pointnet(point_cloud)
        
        # Process image with YOLO backbone
        yolo_output = self.yolo(image)
        image_features = yolo_output['features']
        
        # Fuse features
        fused_features = self.fusion_module(image_features, lidar_features, img_points)
        
        # Generate predictions from fused features
        large_output = self.pred_large(fused_features['large'])
        medium_output = self.pred_medium(fused_features['medium'])
        small_output = self.pred_small(fused_features['small'])
        
        return [large_output, medium_output, small_output]
