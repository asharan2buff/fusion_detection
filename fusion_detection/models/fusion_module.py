# models/fusion_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveFusionBlock(nn.Module):
    def __init__(self, image_feat_dim, lidar_feat_dim, hidden_dim=256, fusion_type='residual'):
        super(AdaptiveFusionBlock, self).__init__()
        self.fusion_type = fusion_type
        
        # Dimension reduction for input features
        self.img_reduction = nn.Conv2d(image_feat_dim, hidden_dim, 1)
        self.lidar_reduction = nn.Conv2d(lidar_feat_dim, hidden_dim, 1)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 2, 1),  # 2 attention weights (for image and lidar)
            nn.Softmax(dim=1)
        )
        
        # Fusion layer
        if fusion_type == 'residual':
            self.fusion = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU()
            )
        elif fusion_type == 'concat':
            self.fusion = nn.Sequential(
                nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU()
            )
        
        # Output projection back to original dimensions
        self.out_proj_img = nn.Conv2d(hidden_dim, image_feat_dim, 1)
        
    def forward(self, img_feat, lidar_feat):
        batch_size = img_feat.size(0)
        
        # Reduce dimensions
        img_reduced = self.img_reduction(img_feat)
        lidar_reduced = self.lidar_reduction(lidar_feat)
        
        # Compute attention weights
        concat_feat = torch.cat([img_reduced, lidar_reduced], dim=1)
        attention_weights = self.attention(concat_feat)
        
        # Apply attention weights
        img_weight = attention_weights[:, 0:1, :, :]
        lidar_weight = attention_weights[:, 1:2, :, :]
        
        weighted_img = img_reduced * img_weight
        weighted_lidar = lidar_reduced * lidar_weight
        
        # Fusion
        if self.fusion_type == 'residual':
            fused = weighted_img + weighted_lidar
            fused = self.fusion(fused)
        elif self.fusion_type == 'concat':
            fused = torch.cat([weighted_img, weighted_lidar], dim=1)
            fused = self.fusion(fused)
        
        # Project back to original dimensions
        output = self.out_proj_img(fused)
        
        # Residual connection for image features
        output = output + img_feat
        
        return output

class SpatialILFusionModule(nn.Module):
    def __init__(self, image_features_dims, lidar_feat_dim=1024, fusion_type='residual'):
        super(SpatialILFusionModule, self).__init__()
        self.fusion_blocks = nn.ModuleDict()
        
        # Create fusion blocks for each feature scale
        for scale, dim in image_features_dims.items():
            self.fusion_blocks[scale] = AdaptiveFusionBlock(
                image_feat_dim=dim,
                lidar_feat_dim=lidar_feat_dim,
                fusion_type=fusion_type
            )
    
    def forward(self, image_features, lidar_features, point_img_coords):
        """
        Args:
            image_features: Dict containing image features at different scales
            lidar_features: Global LiDAR features (B, C)
            point_img_coords: Image coordinates for each LiDAR point (B, N, 2)
        """
        batch_size = lidar_features.size(0)
        fused_features = {}
        
        # Process each scale
        for scale, img_feat in image_features.items():
            B, C, H, W = img_feat.shape
            
            # Create LiDAR feature grid for this scale
            lidar_grid = torch.zeros((B, lidar_features.size(1), H, W), device=img_feat.device)
            
            # For each batch
            for b in range(batch_size):
                # Scale point coordinates to match current feature map
                coords = point_img_coords[b].clone()
                coords[:, 0] = coords[:, 0] * W / 416  # Assuming original image is 416x416
                coords[:, 1] = coords[:, 1] * H / 416
                
                # Quantize coordinates to grid cells
                coords = coords.long()
                
                # Filter out points outside the grid
                valid = (coords[:, 0] >= 0) & (coords[:, 0] < W) & (coords[:, 1] >= 0) & (coords[:, 1] < H)
                valid_coords = coords[valid]
                
                # Place LiDAR features in the grid at corresponding locations
                for idx, (x, y) in enumerate(valid_coords):
                    lidar_grid[b, :, y, x] = lidar_features[b]
            
            # Apply fusion block
            fused_features[scale] = self.fusion_blocks[scale](img_feat, lidar_grid)
        
        return fused_features
