# train.py
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import time
import argparse

from utils.dataset import KittiMultiModalDataset
from models.fusion_detection_model import SpatialILFusionModel

def parse_args():
    parser = argparse.ArgumentParser(description='Train the Spatial-IL Fusion Pipeline')
    parser.add_argument('--config', type=str, default='config/yolo_pointnet_fusion_trainer.json',
                        help='Path to the configuration file')
    return parser.parse_args()

class YOLOLoss(nn.Module):
    def __init__(self, num_classes=8, anchors=None):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        
        # Default anchors for YOLO (can be customized for KITTI)
        if anchors is None:
            self.anchors = [
                [[10, 13], [16, 30], [33, 23]],  # Small
                [[30, 61], [62, 45], [59, 119]],  # Medium
                [[116, 90], [156, 198], [373, 326]]  # Large
            ]
        else:
            self.anchors = anchors
            
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
    def forward(self, predictions, targets, input_dim):
        """
        predictions: List of outputs from the model [large, medium, small]
        targets: Ground truth boxes in format [batch_idx, class, x, y, w, h]
        input_dim: Input dimensions of the image
        """
        if targets.shape[0] == 0:  # No objects in this batch
            return torch.tensor(0.0, requires_grad=True, device=predictions[0].device)
            
        total_loss = 0
        device = predictions[0].device
        
        for i, pred in enumerate(predictions):
            batch_size, _, grid_h, grid_w = pred.shape
            num_anchors = 3
            
            # Reshape prediction to [batch, anchors, grid, grid, xywh + conf + classes]
            prediction = pred.view(batch_size, num_anchors, self.num_classes + 5, grid_h, grid_w)
            prediction = prediction.permute(0, 1, 3, 4, 2).contiguous()
            
            # Get outputs
            x = torch.sigmoid(prediction[..., 0])          # Center x
            y = torch.sigmoid(prediction[..., 1])          # Center y
            w = prediction[..., 2]                         # Width
            h = prediction[..., 3]                         # Height
            pred_conf = torch.sigmoid(prediction[..., 4])  # Objectness confidence
            pred_cls = torch.sigmoid(prediction[..., 5:])  # Class predictions
            
            # Calculate grid offsets
            grid_x = torch.arange(grid_w, dtype=torch.float, device=device).repeat(grid_h, 1)
            grid_y = torch.arange(grid_h, dtype=torch.float, device=device).repeat(grid_w, 1).t()
            
            # Scale anchors to grid
            anchors_tensor = torch.tensor(self.anchors[i], dtype=torch.float, device=device)
            scaled_anchors = anchors_tensor / torch.tensor([input_dim[0] / grid_w, input_dim[1] / grid_h], 
                                                          dtype=torch.float, device=device)
            
            # Add offset and scale with anchors
            anchor_w = scaled_anchors[:, 0:1].view((1, num_anchors, 1, 1))
            anchor_h = scaled_anchors[:, 1:2].view((1, num_anchors, 1, 1))
            
            # Loss tensors
            obj_mask = torch.zeros(batch_size, num_anchors, grid_h, grid_w, dtype=torch.bool, device=device)
            noobj_mask = torch.ones(batch_size, num_anchors, grid_h, grid_w, dtype=torch.bool, device=device)
            tx = torch.zeros(batch_size, num_anchors, grid_h, grid_w, dtype=torch.float, device=device)
            ty = torch.zeros(batch_size, num_anchors, grid_h, grid_w, dtype=torch.float, device=device)
            tw = torch.zeros(batch_size, num_anchors, grid_h, grid_w, dtype=torch.float, device=device)
            th = torch.zeros(batch_size, num_anchors, grid_h, grid_w, dtype=torch.float, device=device)
            tconf = torch.zeros(batch_size, num_anchors, grid_h, grid_w, dtype=torch.float, device=device)
            tcls = torch.zeros(batch_size, num_anchors, grid_h, grid_w, self.num_classes, dtype=torch.float, device=device)
            
            # Process targets
            for b in range(batch_size):
                b_targets = targets[targets[:, 0] == b]
                if b_targets.size(0) == 0:
                    continue
                    
                # Parse targets (class, x, y, w, h)
                target_cls = b_targets[:, 1].long()
                gx = b_targets[:, 2] * grid_w  # grid x
                gy = b_targets[:, 3] * grid_h  # grid y
                gw = b_targets[:, 4] * input_dim[0]  # grid width
                gh = b_targets[:, 5] * input_dim[1]  # grid height
                
                # Find best anchor for each target
                gi = gx.long().clamp(0, grid_w - 1)
                gj = gy.long().clamp(0, grid_h - 1)
                
                for t in range(b_targets.size(0)):
                    # Find best anchor based on IoU
                    target_wh = torch.tensor([gw[t], gh[t]], device=device)
                    anchor_ious = []
                    for anc in scaled_anchors:
                        anc = anc * torch.tensor([grid_w, grid_h], device=device)
                        anchor_wh = torch.tensor([anc[0] * input_dim[0] / grid_w, anc[1] * input_dim[1] / grid_h], device=device)
                        iou = bbox_iou(target_wh.unsqueeze(0), anchor_wh.unsqueeze(0))
                        anchor_ious.append(iou)
                    best_anchor = torch.argmax(torch.tensor(anchor_ious, device=device))
                    
                    # Update masks and targets
                    obj_mask[b, best_anchor, gj[t], gi[t]] = True
                    noobj_mask[b, best_anchor, gj[t], gi[t]] = False
                    
                    # Coordinates
                    tx[b, best_anchor, gj[t], gi[t]] = gx[t] - gi[t]
                    ty[b, best_anchor, gj[t], gi[t]] = gy[t] - gj[t]
                    
                    # Width and height
                    tw[b, best_anchor, gj[t], gi[t]] = torch.log(gw[t] / (scaled_anchors[best_anchor][0] * input_dim[0] / grid_w) + 1e-16)
                    th[b, best_anchor, gj[t], gi[t]] = torch.log(gh[t] / (scaled_anchors[best_anchor][1] * input_dim[1] / grid_h) + 1e-16)
                    
                    # Confidence
                    tconf[b, best_anchor, gj[t], gi[t]] = 1
                    
                    # Class
                    tcls[b, best_anchor, gj[t], gi[t], target_cls[t]] = 1
            
            # Calculate losses
            obj_mask_float = obj_mask.float()
            
            # Box loss (x, y)
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            
            # Box loss (w, h)
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            
            # Confidence loss
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = loss_conf_obj + 0.5 * loss_conf_noobj
            
            # Class loss
            if obj_mask.sum() > 0:
                loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            else:
                loss_cls = torch.tensor(0.0, device=device)
            
            # Total loss for this scale
            layer_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
            total_loss += layer_loss
            
        return total_loss

def bbox_iou(box1, box2):
    """
    Calculate IoU between box1 and box2
    box format: [width, height]
    """
    b1_w, b1_h = box1[:, 0], box1[:, 1]
    b2_w, b2_h = box2[:, 0], box2[:, 1]
    
    # Area of boxes
    b1_area = b1_w * b1_h
    b2_area = b2_w * b2_h
    
    # Find the smallest box that contains both boxes
    inter_area = torch.min(b1_w, b2_w) * torch.min(b1_h, b2_h)
    
    # IoU
    return inter_area / (b1_area + b2_area - inter_area + 1e-16)

def train():
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Create output directory
    output_dir = os.path.join(config['trainer_kwargs']['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    
    # Create tensorboard writer
    writer = SummaryWriter(os.path.join(output_dir, 'logs'))
    
    # Set up datasets
    train_dataset = KittiMultiModalDataset(
        **config['dataset_kwargs']['trainer_dataset_kwargs'],
        robustness_augmentations=config['trainer_kwargs'].get('robustness_augmentations', [])
    )
    
    val_dataset = KittiMultiModalDataset(
        **config['dataset_kwargs']['validation_dataset_kwargs']
    )
    
    # Set up dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['trainer_kwargs']['batch_size'],
        shuffle=True,
        num_workers=2,  # Reduce worker count for macOS
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['trainer_kwargs']['batch_size'],
        shuffle=False,
        num_workers=2,  # Reduce worker count for macOS
        pin_memory=True
    )
    
    # Check if MPS is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device for training")
    else:
        device = torch.device("cpu")
        print("MPS device not found, using CPU for training")
    
    # Set up model
    model = SpatialILFusionModel(
        num_classes=len(config['dataset_kwargs']['trainer_dataset_kwargs']['classes']),
        num_points=config['pointnet_kwargs']['num_points'],
        num_global_feats=config['pointnet_kwargs']['num_global_feats'],
        fusion_type=config['adaptive_fusion_kwargs']['fusion_type']
    )
    
    model = model.to(device)
    
    # Set up optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['trainer_kwargs']['learning_rate'],
        weight_decay=config['trainer_kwargs']['weight_decay']
    )
    
    # Set up learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['trainer_kwargs']['scheduler_step_size'],
        gamma=config['trainer_kwargs']['scheduler_gamma']
    )
    
    # Set up loss function
    criterion = YOLOLoss(num_classes=len(config['dataset_kwargs']['trainer_dataset_kwargs']['classes']))
    
    # Training loop
    best_val_loss = float('inf')
    start_epoch = 0
    
    # Load checkpoint if specified
    if config['trainer_kwargs']['checkpoint_idx'] is not None:
        checkpoint_path = os.path.join(output_dir, f"checkpoint_{config['trainer_kwargs']['checkpoint_idx']}.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Main training loop
    for epoch in range(start_epoch, config['trainer_kwargs']['epochs']):
        # Training phase
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['trainer_kwargs']['epochs']}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            images = batch['image'].to(device)
            point_clouds = batch['point_cloud'].to(device)
            img_points = batch['img_points'].to(device)
            targets = batch['boxes'].to(device)
            
            # Forward pass
            predictions = model(images, point_clouds, img_points)
            
            # Compute loss
            loss = criterion(predictions, targets, config['dataset_kwargs']['trainer_dataset_kwargs']['image_size'])
            
            # Backward pass with gradient accumulation
            loss = loss / config['trainer_kwargs']['gradient_accumulation_steps']
            loss.backward()
            
            if (batch_idx + 1) % config['trainer_kwargs']['gradient_accumulation_steps'] == 0:
                # Gradient clipping
                if config['trainer_kwargs']['gradient_clipping'] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['trainer_kwargs']['gradient_clipping'])
                
                # Update weights
                optimizer.step()
                optimizer.zero_grad()
            
            # Update progress bar
            epoch_loss += loss.item() * config['trainer_kwargs']['gradient_accumulation_steps']
            progress_bar.set_postfix({'loss': epoch_loss / (batch_idx + 1)})
        
        # End of epoch
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Training Loss: {avg_loss:.6f}")
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Validation phase
        if config['trainer_kwargs']['monitor_val'] and epoch >= config['trainer_kwargs']['first_val_epoch']:
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    # Move data to device
                    images = batch['image'].to(device)
                    point_clouds = batch['point_cloud'].to(device)
                    img_points = batch['img_points'].to(device)
                    targets = batch['boxes'].to(device)
                    
                    # Forward pass
                    predictions = model(images, point_clouds, img_points)
                    
                    # Compute loss
                    loss = criterion(predictions, targets, config['dataset_kwargs']['validation_dataset_kwargs']['image_size'])
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss:.6f}")
            
            # Log to tensorboard
            writer.add_scalar('Loss/val', avg_val_loss, epoch)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                }, os.path.join(output_dir, 'best_model.pth'))
                print(f"Saved best model with validation loss: {best_val_loss:.6f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(output_dir, f'checkpoint_{epoch+1}.pth'))
        
        # Update scheduler
        scheduler.step()
    
    writer.close()
    print("Training completed!")

if __name__ == '__main__':
    train()
