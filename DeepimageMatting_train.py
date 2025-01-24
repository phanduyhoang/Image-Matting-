import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F  # For PyTorch functional operations like relu
import torchvision.transforms.functional as TF  # For transformations like vflip, hflip

import matplotlib.pyplot as plt


## you can replace the path below with your image/trimap/ground truth path.
#I wrote the code on kaggle and too lazay to change it. the dataset I used was from alpahmatting.com
# Set directories
DATASET_PATH = "/kaggle/input/alpha-matte-dataset/"
GT_DIR = os.path.join(DATASET_PATH, "gt_training_lowres")
INPUT_DIR = os.path.join(DATASET_PATH, "input_training_lowres")
TRIMAP_DIR = os.path.join(DATASET_PATH, "trimap_training_lowres/Trimap1")

# Fixed image size
IMG_SIZE = (512, 512)

################################################################################
# 1) Custom Dataset
################################################################################
class AlphaMatteDataset(Dataset):
    def __init__(self, input_dir, trimap_dir, gt_dir, transform=None):
        self.input_dir = input_dir
        self.trimap_dir = trimap_dir
        self.gt_dir = gt_dir
        self.input_files = sorted(os.listdir(input_dir))  # Sort to match trimap/gt
        self.transform = transform

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        # Load input, trimap, and ground truth
        input_path = os.path.join(self.input_dir, self.input_files[idx])
        trimap_path = os.path.join(self.trimap_dir, self.input_files[idx])
        gt_path = os.path.join(self.gt_dir, self.input_files[idx])
        
        input_image = cv2.imread(input_path, cv2.IMREAD_COLOR)  # RGB image
        trimap = cv2.imread(trimap_path, cv2.IMREAD_GRAYSCALE)  # Single channel
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)          # Ground truth alpha

        # Convert to RGB and normalize
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        
        # Resize all to fixed size
        input_image = cv2.resize(input_image, IMG_SIZE)
        trimap = cv2.resize(trimap, IMG_SIZE)
        gt = cv2.resize(gt, IMG_SIZE)

        # Convert to tensor
        input_image = transforms.ToTensor()(input_image)
        trimap = transforms.ToTensor()(trimap)
        gt = transforms.ToTensor()(gt)

        # Apply transforms
        if self.transform:
            input_image, trimap, gt = self.transform(input_image, trimap, gt)

        return input_image, trimap, gt

################################################################################
# 2) Transformations (Augmentations)
################################################################################
def random_transform(input_image, trimap, gt):
    # Random horizontal flip
    if random.random() > 0.5:
        input_image = TF.hflip(input_image)
        trimap = TF.hflip(trimap)
        gt = TF.hflip(gt)

    # Random vertical flip
    if random.random() > 0.5:
        input_image = TF.vflip(input_image)
        trimap = TF.vflip(trimap)
        gt = TF.vflip(gt)

    # Color jitter on input image
    color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    input_image = color_jitter(input_image)

    return input_image, trimap, gt


################################################################################
# 3) Loss Functions
################################################################################
def alpha_prediction_loss(pred_alpha, gt_alpha):
    return torch.mean((pred_alpha - gt_alpha) ** 2)

def compositional_loss(pred_alpha, gt_alpha, fg, bg, input_image):
    pred_composite = pred_alpha * fg + (1 - pred_alpha) * bg
    gt_composite = gt_alpha * fg + (1 - gt_alpha) * bg
    return torch.mean((pred_composite - gt_composite) ** 2)

################################################################################
# 4) Training Setup
################################################################################
# Dataset and DataLoader
dataset = AlphaMatteDataset(
    INPUT_DIR,
    TRIMAP_DIR,
    GT_DIR,
    transform=random_transform
)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

# Model, optimizer, and scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TwoStageMattingNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in data_loader:
        input_image, trimap, gt_alpha = batch
        input_image = input_image.to(device)
        trimap = trimap.to(device)
        gt_alpha = gt_alpha.to(device)

        # Forward pass
        raw_alpha, refined_alpha = model(input_image, trimap)

        # Losses
        pred_loss = alpha_prediction_loss(refined_alpha, gt_alpha)
        # Dummy foreground/background for compositional loss
        fg = torch.ones_like(gt_alpha)  # Replace with actual FG
        bg = torch.zeros_like(gt_alpha)  # Replace with actual BG
        comp_loss = compositional_loss(refined_alpha, gt_alpha, fg, bg, input_image)

        loss = pred_loss + comp_loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Scheduler step
    scheduler.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(data_loader):.4f}")

################################################################################
# 5) Save the Model
################################################################################
torch.save(model.state_dict(), "matting_model.pth")
print("Model saved!")
