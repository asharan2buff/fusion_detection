# eval.py
import os
import json
import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from utils.dataset import KittiMultiModalDataset
from models.fusion_detection_model import SpatialILFusionModel

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate the Spatial-IL Fusion Pipeline')
    parser.add_argument('--config', type=str, default='config/yolo_pointnet_fusion_trainer.json',
                        help='Path to the configuration file')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the model checkpoint. If not specified, use the best model from the output directory')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to visualize')
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                        help='Confidence threshold for detection')
    parser.add_argument('--nms_threshold', type=float, default=0.4,
                        help='NMS threshold for detection')
    return parser.parse_args()

def post_process_detections(predictions, image_size, num_classes=8, conf_threshold=0.5, nms_threshold=0.4):
    """
    Apply non-maximum suppression to the YOLO predictions
    """
    # Anchors for YOLO
    anchors = [
        [[10, 13], [16, 30], [33, 23]],  # Small
        [[30, 61], [62, 45], [59, 119]],  # Medium
        [[116, 90], [156, 198], [373, 326]]  # Large
    ]
    
    results = []
    
    # Process each scale
    for i, prediction in enumerate(predictions):
        # Get dimensions
        batch_size, _, height, width = prediction.shape
        num_anchors = 3
        
        # Reshape prediction to [batch, anchors, grid, grid, xywh + conf + classes]
        prediction = prediction.view(batch_size, num_anchors, num_classes + 5, height, width)
        prediction = prediction.permute(0, 1, 3, 4, 2).contiguous()
        
        # Apply sigmoid to confidence and class scores
        prediction[..., 4:] = torch.sigmoid(prediction[..., 4:])
        
        # Process each item in the batch
        for b in range(batch_size):
            boxes = []
            scores = []
            class_ids = []
            
            # Process each anchor
            for a in range(num_anchors):
                # Get confidence scores above threshold
                conf_mask = prediction[b, a, :, :, 4] > conf_threshold
                if not conf_mask.any():
                    continue
                
                # Get only the boxes with confidence > threshold
                masked_pred = prediction[b, a, conf_mask, :]
                
                # Get grid positions for these boxes
                grid_y, grid_x = torch.nonzero(conf_mask, as_tuple=True)
                
                # Get box parameters
                x = (torch.sigmoid(masked_pred[:, 0]) + grid_x.float()) / width
                y = (torch.sigmoid(masked_pred[:, 1]) + grid_y.float()) / height
                w = torch.exp(masked_pred[:, 2]) * anchors[i][a][0] / image_size[0]
                h = torch.exp(masked_pred[:, 3]) * anchors[i][a][1] / image_size[1]
                
                # Convert to corner format (xmin, ymin, xmax, ymax)
                xmin = (x - w / 2).clamp(0, 1) * image_size[0]
                ymin = (y - h / 2).clamp(0, 1) * image_size[1]
                xmax = (x + w / 2).clamp(0, 1) * image_size[0]
                ymax = (y + h / 2).clamp(0, 1) * image_size[1]
                
                # Get confidence scores
                conf = masked_pred[:, 4]
                
                # Get class scores and IDs
                class_scores, class_ids_tensor = torch.max(masked_pred[:, 5:], dim=1)
                
                # Combine confidence with class probability
                final_scores = conf * class_scores
                
                # Convert to numpy
                boxes.extend(torch.stack((xmin, ymin, xmax, ymax), dim=1).cpu().numpy())
                scores.extend(final_scores.cpu().numpy())
                class_ids.extend(class_ids_tensor.cpu().numpy())
            
            # Apply NMS
            if len(boxes) > 0:
                boxes = np.array(boxes)
                scores = np.array(scores)
                class_ids = np.array(class_ids)
                
                # Apply NMS for each class
                final_boxes = []
                final_scores = []
                final_class_ids = []
                
                for c in range(num_classes):
                    class_mask = class_ids == c
                    if not class_mask.any():
                        continue
                    
                    c_boxes = boxes[class_mask]
                    c_scores = scores[class_mask]
                    
                    # Apply NMS
                    keep_indices = nms(c_boxes, c_scores, nms_threshold)
                    
                    final_boxes.extend(c_boxes[keep_indices])
                    final_scores.extend(c_scores[keep_indices])
                    final_class_ids.extend([c] * len(keep_indices))
                
                results.append({
                    'boxes': np.array(final_boxes),
                    'scores': np.array(final_scores),
                    'class_ids': np.array(final_class_ids)
                })
            else:
                results.append({
                    'boxes': np.array([]),
                    'scores': np.array([]),
                    'class_ids': np.array([])
                })
    
    return results

def nms(boxes, scores, threshold):
    """
    Apply non-maximum suppression to boxes
    """
    if len(boxes) == 0:
        return []
    
    # Convert to float
    boxes = boxes.astype(float)
    
    # Get coordinates
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # Calculate area
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # Sort by confidence
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # Compute IoU
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        
        intersection = w * h
        iou = intersection / (area[i] + area[order[1:]] - intersection)
        
        # Remove boxes with IoU > threshold
        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]
    
    return keep

def visualize_detections(image, detections, classes, sample_id, output_dir):
    """
    Visualize detections on the image
    """
    # Copy image for visualization
    vis_image = image.copy()
    
    # Colors for different classes
    colors = plt.cm.hsv(np.linspace(0, 1, len(classes)))
    colors = (colors[:, :3] * 255).astype(int)
    
    # Draw boxes
    for i, box in enumerate(detections['boxes']):
        # Get box coordinates
        x1, y1, x2, y2 = box.astype(int)
        
        # Get class and confidence
        class_id = detections['class_ids'][i]
        score = detections['scores'][i]
        
        # Get color for class
        color = tuple(map(int, colors[class_id]))
        
        # Draw rectangle
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{classes[class_id]}: {score:.2f}"
        cv2.putText(vis_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Save visualization
    cv2.imwrite(os.path.join(output_dir, f"{sample_id}_detection.jpg"), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    
    return vis_image

def evaluate():
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Get output directory
    output_dir = os.path.join(config['trainer_kwargs']['output_dir'], 'eval_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up validation dataset
    val_dataset = KittiMultiModalDataset(
        **config['dataset_kwargs']['validation_dataset_kwargs']
    )
    
    # Set up dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Process one sample at a time for visualization
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Check if MPS is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device for evaluation")
    else:
        device = torch.device("cpu")
        print("MPS device not found, using CPU for evaluation")
    
    # Set up model
    model = SpatialILFusionModel(
        num_classes=len(config['dataset_kwargs']['validation_dataset_kwargs']['classes']),
        num_points=config['pointnet_kwargs']['num_points'],
        num_global_feats=config['pointnet_kwargs']['num_global_feats'],
        fusion_type=config['adaptive_fusion_kwargs']['fusion_type']
    )
    
    model = model.to(device)
    
    # Load model
    if args.model_path:
        checkpoint_path = args.model_path
    else:
        checkpoint_path = os.path.join(config['trainer_kwargs']['output_dir'], 'best_model.pth')
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {checkpoint_path}")
    else:
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        return
    
    # Set model to evaluation mode
    model.eval()
    
    # Get class names
    classes = config['dataset_kwargs']['validation_dataset_kwargs']['classes']
    
    # Evaluation loop
    processed_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
            if processed_samples >= args.num_samples:
                break
            
            # Get sample ID
            sample_id = batch['image_id'][0]
            
            # Move data to device
            images = batch['image'].to(device)
            point_clouds = batch['point_cloud'].to(device)
            img_points = batch['img_points'].to(device)
            
            # Original image size
            orig_h, orig_w = batch['orig_size']
            
            # Forward pass
            predictions = model(images, point_clouds, img_points)
            
            # Post-process detections
            detections = post_process_detections(
                predictions, 
                config['dataset_kwargs']['validation_dataset_kwargs']['image_size'],
                len(classes),
                args.conf_threshold,
                args.nms_threshold
            )[0]  # Get first batch item
            
            # Convert image back to numpy
            image = (images[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            # Resize to original size for visualization
            image = cv2.resize(image, (orig_w, orig_h))
            
            # Visualize detections
            vis_image = visualize_detections(image, detections, classes, sample_id, output_dir)
            
            processed_samples += 1
    
    print(f"Evaluation complete. Results saved to {output_dir}")

if __name__ == '__main__':
    evaluate()
