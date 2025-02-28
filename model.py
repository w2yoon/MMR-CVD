# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class CFPFeatureExtractor(nn.Module):
    def __init__(self, fine_tune=True):
        super(CFPFeatureExtractor, self).__init__()
        self.convnext = timm.create_model('convnext_base.fb_in22k_ft_in1k_384', pretrained=True)
        self.convnext.head.fc = nn.Identity()
        self.fc = nn.Linear(self.convnext.num_features, 256)

        if not fine_tune:
            for param in self.convnext.parameters():
                param.requires_grad = False
        else:
            for name, param in self.convnext.named_parameters():
                if "stages.3" in name or "head" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def forward(self, x):
        x = self.convnext(x)
        x = self.fc(x)
        return x

class NonLocalBlock3D(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=False, bn_layer=True):
        super(NonLocalBlock3D, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels or max(in_channels // 2, 1)
        
        self.theta = nn.Conv3d(in_channels, self.inter_channels, kernel_size=1)
        self.phi   = nn.Conv3d(in_channels, self.inter_channels, kernel_size=1)
        self.g     = nn.Conv3d(in_channels, self.inter_channels, kernel_size=1)
        
        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv3d(self.inter_channels, in_channels, kernel_size=1),
                nn.BatchNorm3d(in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv3d(self.inter_channels, in_channels, kernel_size=1)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)
        
        self.sub_sample = sub_sample
        if sub_sample:
            self.max_pool = nn.MaxPool3d(kernel_size=(1,2,2))
    
    def forward(self, x):
        batch_size = x.size(0)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        y = torch.matmul(f_div_C, g_x).permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z

class OCTFeatureExtractor_DilatedAttention(nn.Module):
    def __init__(self, in_channels=1, feature_dim=256, dropout_rate=0.3):
        super(OCTFeatureExtractor_DilatedAttention, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn1   = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.bn2   = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=4, dilation=4)
        self.bn3   = nn.BatchNorm3d(128)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn4   = nn.BatchNorm3d(256)
        
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.attention = NonLocalBlock3D(256)
        self.global_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Linear(256, feature_dim)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  
        x = self.pool(x)                    
        x = F.relu(self.bn2(self.conv2(x)))  
        x = self.pool(x)                    
        x = F.relu(self.bn3(self.conv3(x)))  
        x = self.pool(x)                    
        x = F.relu(self.bn4(self.conv4(x)))  
        x = self.pool(x)                    
        x = self.attention(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        feature = self.fc(x)
        return feature

class CrossAttentionFusion(nn.Module):
    def __init__(self, cfp_dim, oct_dim, fusion_dim, num_heads=4):
        super(CrossAttentionFusion, self).__init__()
        self.project_cfp = nn.Linear(cfp_dim, fusion_dim)
        self.project_oct    = nn.Linear(oct_dim, fusion_dim)
        
        self.cross_attn_cfp = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=num_heads, batch_first=True)
        self.cross_attn_oct    = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=num_heads, batch_first=True)
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim)
        )
    
    def forward(self, cfp_feat, oct_feat):
        cfp_token = self.project_cfp(cfp_feat).unsqueeze(1)
        oct_token    = self.project_oct(oct_feat).unsqueeze(1)
        
        attn_out_cfp, _ = self.cross_attn_cfp(query=cfp_token, key=oct_token, value=oct_token)
        attn_out_oct, _ = self.cross_attn_oct(query=oct_token, key=cfp_token, value=cfp_token)
        
        fused = torch.cat([attn_out_cfp.squeeze(1), attn_out_oct.squeeze(1)], dim=-1)
        fused = self.fusion_mlp(fused)
        return fused

class CFPOCTFusionNet(nn.Module):
    def __init__(self, num_classes=5, fusion_dim=512):
        super(CFPOCTFusionNet, self).__init__()
        self.cfp_backbone = CFPFeatureExtractor(fine_tune=True)
        self.cfp_feat_dim = 256
        
        self.oct_backbone = OCTFeatureExtractor_DilatedAttention(in_channels=1, feature_dim=256)
        self.oct_feat_dim = 256
        
        self.fusion = CrossAttentionFusion(
            cfp_dim=self.cfp_feat_dim,
            oct_dim=self.oct_feat_dim,
            fusion_dim=fusion_dim,
            num_heads=4
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),  
            nn.ReLU(),
            nn.Dropout(0.5),  
            nn.Linear(128, 128), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, cfp, oct_volume):
        cfp_feat = self.cfp_backbone(cfp)
        oct_feat    = self.oct_backbone(oct_volume)
        fused_feat  = self.fusion(cfp_feat, oct_feat)
        output = self.classifier(fused_feat)
        return output
