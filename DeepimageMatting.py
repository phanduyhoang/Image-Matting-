#Source paper
#https://arxiv.org/pdf/1703.03872

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

################################################################################
# 1) VGG-16 Encoder (blocks 1..4) with 4-channel input
#    We collect 3 skip outputs (from the ends of blocks 1,2,3),
#    then do block4 + "fc6-as-conv" as the bottom (1/16 scale).
################################################################################
class VGG16Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Load VGG16 with new "weights" param (instead of pretrained=True)
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        features = vgg16.features  # nn.Sequential of conv/pool layers

        # 1) Modify the first Conv2d to take 4 channels (RGB+Trimap)
        old_conv0 = features[0]  # conv2d(3,64, kernel_size=3, padding=1)
        new_conv0 = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        with torch.no_grad():
            # Copy weights for RGB, set the 4th channel to zero
            new_conv0.weight[:, :3] = old_conv0.weight
            new_conv0.weight[:, 3]  = 0.0
            new_conv0.bias[:]       = old_conv0.bias
        features[0] = new_conv0
        
        # 2) Split the standard VGG16 feature extractor into four blocks:
        #
        #    block1: [0..4]   = conv1_1, relu, conv1_2, relu, pool => 1/2 scale
        #    block2: [5..9]   = conv2_1, relu, conv2_2, relu, pool => 1/4 scale
        #    block3: [10..16] = conv3_1, relu, conv3_2, relu, conv3_3, relu, pool => 1/8 scale
        #    block4: [17..23] = conv4_1, relu, conv4_2, relu, conv4_3, relu, pool => 1/16 scale
        #
        #    (We won't use block5 for this UNet-style matting.)
        #
        self.block1 = features[0:5]    # conv1_x + pool => output 1/2
        self.block2 = features[5:10]   # conv2_x + pool => output 1/4
        self.block3 = features[10:17]  # conv3_x + pool => output 1/8
        self.block4 = features[17:24]  # conv4_x + pool => output 1/16

        # 3) Turn "fc6" into a 1×1 or 3×3 conv.  In official VGG, fc6 is 4096 units.
        #    We can reduce it here to keep memory smaller. E.g. out_channels=512.
        #    We'll use kernel_size=3, stride=1, padding=1 so shape stays 1/16.
        self.fc6_conv = nn.Conv2d(in_channels=512, out_channels=512,
                                  kernel_size=3, padding=1)
        # Initialize fc6 from the old VGG fc6 if you like. We do a fresh Xavier init:
        nn.init.xavier_normal_(self.fc6_conv.weight)

    def forward(self, x):
        """
        x: [B,4,H,W]  (RGB + Trimap)
        Returns:
          bottom:  [B,512, H/16, W/16] after block4 + fc6
          skips: [skip1, skip2, skip3]
                  skip1 => [B,64,  H/2,  W/2]
                  skip2 => [B,128, H/4,  W/4]
                  skip3 => [B,256, H/8,  W/8]
        """
        # Block1 => 1/2 scale
        x = self.block1(x)
        skip1 = x  # shape ~ [B,64, H/2, W/2]

        # Block2 => 1/4 scale
        x = self.block2(x)
        skip2 = x  # shape ~ [B,128, H/4, W/4]

        # Block3 => 1/8 scale
        x = self.block3(x)
        skip3 = x  # shape ~ [B,256, H/8, W/8]

        # Block4 => 1/16 scale
        x = self.block4(x)  # shape ~ [B,512, H/16, W/16]

        # fc6-as-conv => keep 1/16 scale
        x = self.fc6_conv(x)  # [B,512, H/16, W/16]
        x = F.relu(x, inplace=True)

        return x, [skip1, skip2, skip3]

################################################################################
# 2) UNet-Style Decoder: from 1/16 -> 1/8 -> 1/4 -> 1/2 -> 1
################################################################################
class DecoderBlock(nn.Module):
    """
    A simple "cat->conv->conv->relu" block, for after we upsample.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class MattingDecoder(nn.Module):
    """
    Decode from bottom feature (1/16 scale) back to full res 1× size.
    We'll do 3 "decoder steps" merging skip3, skip2, skip1, then
    a final upsample to 1× and 1×1 conv => alpha.
    """
    def __init__(self):
        super().__init__()
        # After block4+fc6, we have [B,512,H/16,W/16].
        # skip3 => [B,256,H/8,W/8]
        # skip2 => [B,128,H/4,W/4]
        # skip1 => [B, 64,H/2,W/2]

        # Step 1: Upsample from 1/16 => 1/8, cat skip3 => out => [B,256?]
        self.dec3 = DecoderBlock(in_ch=512+256, out_ch=256)
        # Step 2: Upsample from 1/8 => 1/4, cat skip2 => out => [B,128?]
        self.dec2 = DecoderBlock(in_ch=256+128, out_ch=128)
        # Step 3: Upsample from 1/4 => 1/2, cat skip1 => out => [B,64?]
        self.dec1 = DecoderBlock(in_ch=128+64, out_ch=64)

        # Finally upsample 1/2 => 1, then do 1×1 conv => 1 channel alpha
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, bottom, skips):
        """
        bottom: [B,512,H/16,W/16]
        skips: [skip1, skip2, skip3]
        """
        skip1, skip2, skip3 = skips

        # 1) upsample bottom 1/16 -> 1/8
        x = F.interpolate(bottom, scale_factor=2, mode='bilinear', align_corners=True)
        # cat skip3 (which is 1/8 scale)
        x = torch.cat([x, skip3], dim=1)  # => [B,512+256, H/8, W/8]
        x = self.dec3(x)  # => [B,256, H/8, W/8]

        # 2) upsample 1/8 -> 1/4
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, skip2], dim=1)  # => [B,256+128, H/4, W/4]
        x = self.dec2(x)  # => [B,128, H/4, W/4]

        # 3) upsample 1/4 -> 1/2
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, skip1], dim=1)  # => [B,128+64, H/2, W/2]
        x = self.dec1(x)  # => [B,64, H/2, W/2]

        # 4) final upsample 1/2 -> 1
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)  
        # => [B,64,H,W]

        alpha = self.final_conv(x)  # => [B,1,H,W]
        return alpha

################################################################################
# 3) Stage-1 Matting Network:  (VGG16 Encoder -> UNet Decoder)
################################################################################
class Stage1MattingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VGG16Encoder()
        self.decoder = MattingDecoder()

    def forward(self, x):
        """
        x: [B,4,H,W] => raw_alpha [B,1,H,W]
        """
        bottom, skips = self.encoder(x)
        alpha = self.decoder(bottom, skips)
        return alpha

################################################################################
# 4) Refinement Network: (raw_alpha + original RGB) -> refined_alpha
################################################################################
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)
    def forward(self, x):
        skip = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + skip
        x = self.relu(x)
        return x

class RefinementNet(nn.Module):
    """
    Takes [B,4,H,W] = (3-channel RGB + 1-channel raw alpha),
    outputs refined alpha of shape [B,1,H,W].
    """
    def __init__(self, in_ch=4, mid_ch=64):
        super().__init__()
        self.head = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1)
        self.bn_head = nn.BatchNorm2d(mid_ch)
        self.res1 = ResidualBlock(mid_ch)
        self.res2 = ResidualBlock(mid_ch)
        self.res3 = ResidualBlock(mid_ch)
        self.final_conv = nn.Conv2d(mid_ch, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn_head(self.head(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        out_alpha = self.final_conv(x)
        return out_alpha

################################################################################
# 5) Two-Stage Model (Stage1 + Stage2)
################################################################################
class TwoStageMattingNet(nn.Module):
    """
    - Stage1: UNet (VGG16) => raw_alpha
    - Stage2: refine => refined_alpha
    """
    def __init__(self):
        super().__init__()
        self.stage1 = Stage1MattingNet()
        self.stage2 = RefinementNet()

    def forward(self, rgb, trimap):
        """
        rgb:    [B,3,H,W]
        trimap: [B,1,H,W]
        Returns (raw_alpha, refined_alpha)
        """
        # 1) Stage1
        x_stage1 = torch.cat([rgb, trimap], dim=1)  # => [B,4,H,W]
        raw_alpha = self.stage1(x_stage1)           # => [B,1,H,W]

        # 2) Stage2
        x_stage2 = torch.cat([rgb, raw_alpha], dim=1)  # => [B,4,H,W]
        refined_alpha = self.stage2(x_stage2)           # => [B,1,H,W]

        return raw_alpha, refined_alpha

################################################################################
# 6) Demo / Test
################################################################################
if __name__ == "__main__":
    B, H, W = 1, 512, 512
    rgb    = torch.randn(B, 3, H, W)
    trimap = torch.randn(B, 1, H, W)

    model = TwoStageMattingNet()
    with torch.no_grad():
        raw_alpha, refined_alpha = model(rgb, trimap)

    print("rgb:           ", rgb.shape)
    print("trimap:        ", trimap.shape)
    print("raw_alpha:     ", raw_alpha.shape)
    print("refined_alpha: ", refined_alpha.shape)
