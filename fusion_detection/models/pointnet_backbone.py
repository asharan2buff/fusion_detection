# models/pointnet_backbone.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
    def forward(self, x):
        batch_size = x.size()[0]
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        # Initialize as identity matrix
        identity = torch.eye(self.k, requires_grad=True, device=x.device)
        identity = identity.repeat(batch_size, 1, 1)
            
        x = x.view(-1, self.k, self.k) + identity
        return x

class PointNetBackbone(nn.Module):
    def __init__(self, num_points=2500, num_global_feats=1024, input_channels=4):
        super(PointNetBackbone, self).__init__()
        self.num_points = num_points
        self.input_transform = TNet(k=input_channels)
        self.feature_transform = TNet(k=64)
        
        self.conv1 = nn.Conv1d(input_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, num_global_feats, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(num_global_feats)
        
    def forward(self, x):
        # x shape: (batch_size, num_points, input_channels)
        x = x.transpose(2, 1)  # (batch_size, input_channels, num_points)
        
        # Input transformation
        trans = self.input_transform(x)
        x = torch.bmm(x.transpose(2, 1), trans).transpose(2, 1)
        
        # MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Feature transformation
        trans_feat = self.feature_transform(x)
        x = torch.bmm(x.transpose(2, 1), trans_feat).transpose(2, 1)
        
        # More MLPs
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))
        
        # Global feature vector
        global_feature = torch.max(x, 2, keepdim=True)[0]  # (batch_size, num_global_feats, 1)
        
        return global_feature.squeeze(2), trans_feat
