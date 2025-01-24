import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101
import torchvision

# Helper function to replace BatchNorm2d with GroupNorm
def replace_bn_with_gn(module, num_groups=32):
    """
    Recursively replace all nn.BatchNorm2d layers with nn.GroupNorm in a module.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            # Ensure num_channels is divisible by num_groups
            if num_channels % num_groups != 0:
                # Find the largest divisor of num_channels less than or equal to num_groups
                for ng in range(num_groups, 0, -1):
                    if num_channels % ng == 0:
                        num_groups = ng
                        break
            # Replace BatchNorm2d with GroupNorm
            setattr(module, name, nn.GroupNorm(num_groups=num_groups, num_channels=num_channels))
        else:
            replace_bn_with_gn(child, num_groups=num_groups)

# Atrous Convolution with GroupNorm
class AtrousConv(nn.Module):
    def __init__(self, in_channels, out_channels, rate, num_groups=32):
        super(AtrousConv, self).__init__()
        self.atrous = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False
        )
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.gn(self.atrous(x)))

# Atrous Spatial Pyramid Pooling (ASPP) Module with GroupNorm
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=32):
        super(ASPP, self).__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU()
        )
        self.conv3x3_r6 = AtrousConv(in_channels, out_channels, rate=6, num_groups=num_groups)
        self.conv3x3_r12 = AtrousConv(in_channels, out_channels, rate=12, num_groups=num_groups)
        self.conv3x3_r18 = AtrousConv(in_channels, out_channels, rate=18, num_groups=num_groups)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU()
        )
        self.concat_conv = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        size = x.shape[-2:]
        conv1x1 = self.conv1x1(x)
        conv3x3_r6 = self.conv3x3_r6(x)
        conv3x3_r12 = self.conv3x3_r12(x)
        conv3x3_r18 = self.conv3x3_r18(x)
        global_avg_pool = F.interpolate(
            self.global_avg_pool(x), size=size, mode='bilinear', align_corners=False
        )
        concatenated = torch.cat([conv1x1, conv3x3_r6, conv3x3_r12, conv3x3_r18, global_avg_pool], dim=1)
        return self.concat_conv(concatenated)

# DeepLabv3 Encoder with GroupNorm and Debugging
class DeepLabv3Encoder(nn.Module):
    def __init__(self, output_stride=16, num_groups=32):
        super(DeepLabv3Encoder, self).__init__()
        # Determine dilation rates based on output_stride
        if output_stride == 16:
            replace_stride_with_dilation = [False, True, True]
        elif output_stride == 8:
            replace_stride_with_dilation = [False, False, True]
        else:
            raise ValueError("Output stride must be either 8 or 16")
    
        # Load ResNet-101 backbone with dilated convolutions
        resnet = resnet101(
            weights=torchvision.models.ResNet101_Weights.DEFAULT,
            replace_stride_with_dilation=replace_stride_with_dilation
        )
        # Replace all BatchNorm2d with GroupNorm
        replace_bn_with_gn(resnet, num_groups=num_groups)
        
        self.conv1 = resnet.conv1
        self.gn1 = resnet.bn1  # Now GroupNorm
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # Block1
        self.layer2 = resnet.layer2  # Block2
        self.layer3 = resnet.layer3  # Block3 with dilation
        self.layer4 = resnet.layer4  # Block4 with dilation
    
        # ASPP Module
        self.aspp = ASPP(2048, 256, num_groups=num_groups)
    
    def forward(self, x):
        print(f"Input shape: {x.shape}")
        # Backbone feature extraction
        x = self.conv1(x)
        print(f"After conv1: {x.shape}")
        x = self.gn1(x)  # GroupNorm
        x = self.relu(x)
        x = self.maxpool(x)
        print(f"After maxpool: {x.shape}")
        
        low_level_features = self.layer1(x)  # Low-level features for decoder
        print(f"After layer1 (low_level_features): {low_level_features.shape}")
        
        x = self.layer2(low_level_features)
        print(f"After layer2: {x.shape}")
        x = self.layer3(x)
        print(f"After layer3: {x.shape}")
        x = self.layer4(x)
        print(f"After layer4: {x.shape}")
        
        # ASPP Module
        x = self.aspp(x)
        print(f"After ASPP: {x.shape}")
    
        return x, low_level_features  # Return high-level features and low-level features for the decoder

# DeepLabv3 Decoder with GroupNorm and Debugging
class DeepLabv3Decoder(nn.Module):
    def __init__(self, low_level_in_channels=256, low_level_out_channels=48, num_classes=21, num_groups_decoder=16):
        super(DeepLabv3Decoder, self).__init__()
        # 1x1 convolution to reduce low-level features channels
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_in_channels, low_level_out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=num_groups_decoder, num_channels=low_level_out_channels),
            nn.ReLU()
        )
        
        # Concatenation and refinement
        self.concat_conv = nn.Sequential(
            nn.Conv2d(256 + low_level_out_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups_decoder, num_channels=256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups_decoder, num_channels=256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
    
    def forward(self, high_level_features, low_level_features):
        low_level = self.low_level_conv(low_level_features)
        print(f"After low_level_conv: {low_level.shape}")
        # Upsample high-level features to low-level spatial dimensions
        high_level_upsampled = F.interpolate(high_level_features, size=low_level.shape[2:], mode='bilinear', align_corners=False)
        print(f"After upsampling high_level_features: {high_level_upsampled.shape}")
        # Concatenate
        concatenated = torch.cat([high_level_upsampled, low_level], dim=1)
        print(f"After concatenation: {concatenated.shape}")
        # Apply convolutions
        out = self.concat_conv(concatenated)
        print(f"After concat_conv: {out.shape}")
        return out

# Complete DeepLabv3 Model Integrating Encoder and Decoder with Debugging
class DeepLabv3(nn.Module):
    def __init__(self, num_classes=21, output_stride=16, num_groups_encoder=32, num_groups_decoder=16):
        super(DeepLabv3, self).__init__()
        self.encoder = DeepLabv3Encoder(output_stride=output_stride, num_groups=num_groups_encoder)
        self.decoder = DeepLabv3Decoder(low_level_in_channels=256, low_level_out_channels=48, num_classes=num_classes, num_groups_decoder=num_groups_decoder)
    
    def forward(self, x):
        high_level_features, low_level_features = self.encoder(x)
        out = self.decoder(high_level_features, low_level_features)
        # Upsample to input image size
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        print(f"Final output shape: {out.shape}")
        return out

# Usage Example
if __name__ == "__main__":
    # Initialize model with separate num_groups for encoder and decoder
    model = DeepLabv3(num_classes=21, output_stride=16, num_groups_encoder=32, num_groups_decoder=16)
    model.train()  # Set to training mode; GroupNorm is compatible with batch size 1
    
    input_tensor = torch.randn(1, 3, 512, 512)  # Batch of 1, RGB image 512x512
    try:
        output = model(input_tensor)
        print("Output shape:", output.shape)  # Expected: (1, 21, 512, 512)
    except Exception as e:
        print("Error during model forward pass:", e)
