#!/usr/bin/env python3
"""
Image Complexity Analyzer v1.1
Created by Philip Hofmann using DeepSeek AI Assistant
Based on ICNet implementation and using model from: https://github.com/tinglyfeng/IC9600
"""

import argparse
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torchvision import transforms
from PIL import Image, ImageFile, UnidentifiedImageError
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import logging
import time
import traceback

# Enable truncated image loading
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ============= Model Architecture =============
class slam(nn.Module):
    """Spatial Attention Module for feature refinement"""
    def __init__(self, spatial_dim):
        super(slam, self).__init__()
        self.spatial_dim = spatial_dim
        self.linear = nn.Sequential(
            nn.Linear(spatial_dim**2, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, feature):
        n, c, h, w = feature.shape
        if h != self.spatial_dim:
            x = F.interpolate(feature, size=(self.spatial_dim, self.spatial_dim), 
                             mode="bilinear", align_corners=True)
        else:
            x = feature

        x = x.view(n, c, -1)
        x = self.linear(x)
        x = x.unsqueeze(dim=3)
        return x.expand_as(feature) * feature

class to_map(nn.Module):
    """Converts features to complexity map"""
    def __init__(self, channels):
        super(to_map, self).__init__()
        self.to_map = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        
    def forward(self, feature):
        return self.to_map(feature)

class conv_bn_relu(nn.Module):
    """Convolution block with batch normalization and ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                              kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class up_conv_bn_relu(nn.Module):
    """Upsampling block with convolution, batch norm and ReLU"""
    def __init__(self, up_size, in_channels, out_channels=64, kernel_size=1, padding=0, stride=1):
        super(up_conv_bn_relu, self).__init__()
        self.upSample = nn.Upsample(size=(up_size, up_size), mode="bilinear", align_corners=True)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.act = nn.ReLU()
        
    def forward(self, x):
        return self.act(self.bn(self.conv(self.upSample(x))))

class ICNet(nn.Module):
    """Image Complexity Network (ICNet) implementation"""
    def __init__(self, is_pretrain=True, size1=512, size2=256):
        super(ICNet, self).__init__()
        # Load pretrained ResNet18 models
        resnet18Pretrained1 = torchvision.models.resnet18(pretrained=is_pretrain)
        resnet18Pretrained2 = torchvision.models.resnet18(pretrained=is_pretrain)
        
        self.size1 = size1
        self.size2 = size2
        
        # Detail branch (high resolution)
        self.b1_1 = nn.Sequential(*list(resnet18Pretrained1.children())[:5])
        self.b1_1_slam = slam(32)
        self.b1_2 = list(resnet18Pretrained1.children())[5]
        self.b1_2_slam = slam(32)

        # Context branch (low resolution)
        self.b2_1 = nn.Sequential(*list(resnet18Pretrained2.children())[:5])
        self.b2_1_slam = slam(32)
        self.b2_2 = list(resnet18Pretrained2.children())[5]
        self.b2_2_slam = slam(32)
        self.b2_3 = list(resnet18Pretrained2.children())[6]
        self.b2_3_slam = slam(16)
        self.b2_4 = list(resnet18Pretrained2.children())[7]
        self.b2_4_slam = slam(8)

        # Upsampling layers
        self.upsize = size1 // 8
        self.up1 = up_conv_bn_relu(up_size=self.upsize, in_channels=128, out_channels=256)
        self.up2 = up_conv_bn_relu(up_size=self.upsize, in_channels=512, out_channels=256)
        
        # Map prediction head
        self.to_map_f = conv_bn_relu(256*2, 256*2)
        self.to_map_f_slam = slam(32)
        self.to_map = to_map(256*2)
        
        # Score prediction head
        self.to_score_f = conv_bn_relu(256*2, 256*2)
        self.to_score_f_slam = slam(32)
        self.head = nn.Sequential(
            nn.Linear(256*2, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
    def forward(self, x1):
        # Verify input size and channels
        if x1.shape[1] != 3:
            raise ValueError(f"Input must have 3 channels, got {x1.shape[1]} channels")
        if x1.shape[2] != self.size1 or x1.shape[3] != self.size1:
            raise ValueError(f"Input size must be ({self.size1}, {self.size1}), got ({x1.shape[2]}, {x1.shape[3]})")
        
        # Process through detail branch
        x1_out = self.b1_1(x1)
        x1_out = self.b1_1_slam(x1_out)
        x1_out = self.b1_2(x1_out)
        x1_out = self.b1_2_slam(x1_out)
        
        # Process through context branch
        x2 = F.interpolate(x1, size=(self.size2, self.size2), mode="bilinear", align_corners=True)
        x2 = self.b2_1(x2)
        x2 = self.b2_1_slam(x2)
        x2 = self.b2_2(x2)
        x2 = self.b2_2_slam(x2)
        x2 = self.b2_3(x2)
        x2 = self.b2_3_slam(x2)
        x2 = self.b2_4(x2)
        x2 = self.b2_4_slam(x2)
        
        # Upsample and concatenate features
        x1_out = self.up1(x1_out)
        x2 = self.up2(x2)
        x_cat = torch.cat((x1_out, x2), dim=1)
        
        # Generate complexity map
        map_feat = self.to_map_f(x_cat)
        map_feat = self.to_map_f_slam(map_feat)
        cly_map = self.to_map(map_feat)
        
        # Generate complexity score
        score_feat = self.to_score_f(x_cat)
        score_feat = self.to_score_f_slam(score_feat)
        score_feat = self.avgpool(score_feat).squeeze()
        score = self.head(score_feat).squeeze()
        
        return score, cly_map

# ============= Main Script =============
def setup_logger(log_path=None):
    """Configure logging to file and console"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Console handler (always active)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(console_handler)
    
    # File handler (only if log_path is provided)
    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
    
    return logger

def parse_arguments():
    """Parse and validate command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Image Complexity Analyzer - Calculate complexity scores for images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', '--input', required=True,
                        help='Input image file or directory')
    parser.add_argument('-o', '--output_csv', required=True,
                        help='Output CSV file path for complexity scores')
    parser.add_argument('-m', '--model', default='complexity.pth',
                        help='Path to model weights file')
    parser.add_argument('-d', '--device', type=int, default=0,
                        help='GPU device index (-1 for CPU)')
    parser.add_argument('-b', '--batch_size', type=int, default=8,
                        help='Batch size for processing')
    parser.add_argument('--save_maps', action='store_true',
                        help='Save complexity maps as .npy files')
    parser.add_argument('--save_visualizations', action='store_true',
                        help='Save visualizations as .png files')
    parser.add_argument('--log_file', default=None,
                        help='Enable logging to specified file (optional)')
    parser.add_argument('--skip_errors', action='store_true',
                        help='Skip images that cause errors')
    
    args = parser.parse_args()
    
    # Set default model path if not provided
    if not os.path.exists(args.model):
        new_path = os.path.join(os.path.dirname(__file__), args.model)
        if os.path.exists(new_path):
            args.model = new_path
        else:
            raise FileNotFoundError(f"Model file not found: {args.model}")
    
    return args

def blend(ori_img, ic_img, alpha=0.8, cmap='magma'):
    """Blend original image with complexity heatmap"""
    cm = plt.get_cmap(cmap)
    cm_ic_map = cm(ic_img) 
    heatmap = Image.fromarray((cm_ic_map[:, :, :3] * 255).astype(np.uint8))
    ori_img = Image.fromarray(ori_img)
    return np.array(Image.blend(ori_img, heatmap, alpha=alpha))

def validate_image(img_path):
    """Robust image validation with multiple checks"""
    try:
        # File accessibility check
        if not os.access(img_path, os.R_OK):
            return False, f"File not readable: {img_path}"
            
        # File size validation
        if os.path.getsize(img_path) == 0:
            return False, f"Empty file: {img_path}"
            
        # Image header validation
        with open(img_path, 'rb') as f:
            header = f.read(24)
            valid_headers = (b'\xff\xd8', b'\x89PNG', b'GIF', b'BM', b'II*\x00', b'MM\x00*')
            if not any(header.startswith(h) for h in valid_headers):
                return False, f"Invalid image header: {img_path}"
                
        # Pillow-based validation
        with Image.open(img_path) as img:
            # Basic attribute check
            if not all(hasattr(img, attr) for attr in ['width', 'height', 'mode']):
                return False, f"Invalid image attributes: {img_path}"
                
            # Thumbnail processing test
            img.thumbnail((16, 16))
            
            # Color conversion test
            if img.mode != 'RGB':
                img.convert('RGB')
                
        return True, ""
    except (IOError, OSError, UnidentifiedImageError) as e:
        return False, f"Image validation failed: {str(e)}"
    except Exception as e:
        return False, f"Unexpected validation error: {str(e)}"

def load_images(image_paths, transform):
    """Load and preprocess a batch of images"""
    batch = []
    valid_paths = []
    original_sizes = []
    skipped = 0
    
    for img_path in image_paths:
        # Validate image before processing
        valid, msg = validate_image(img_path)
        if not valid:
            logging.warning(msg)
            skipped += 1
            continue
            
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                original_sizes.append((img.width, img.height))
                img_tensor = transform(img)
                
                # Verify tensor shape
                if img_tensor.shape[0] != 3:
                    raise ValueError(f"Invalid channel count: {img_tensor.shape[0]} (expected 3)")
                
                batch.append(img_tensor)
                valid_paths.append(img_path)
        except Exception as e:
            logging.error(f"Error loading {img_path}: {str(e)}")
            skipped += 1
            
    if batch:
        return torch.stack(batch), valid_paths, original_sizes, skipped
    return None, [], [], skipped

def process_batch(image_paths, model, device, transform, args, vis_dir):
    """Process a batch of images through the model"""
    # Load and validate images
    batch_tensor, valid_paths, original_sizes, skipped = load_images(image_paths, transform)
    if batch_tensor is None:
        return skipped, 0
    
    try:
        # Move batch to device
        batch_tensor = batch_tensor.to(device)
        
        # Verify input tensor shape
        logging.debug(f"Batch tensor shape: {batch_tensor.shape}")
        if batch_tensor.shape[1] != 3:
            raise ValueError(f"Input tensor must have 3 channels, got {batch_tensor.shape[1]}")
        
        # Run inference
        with torch.no_grad():
            scores, maps = model(batch_tensor)
        
        # Process each image in batch
        processed = 0
        for i, img_path in enumerate(valid_paths):
            try:
                # Save complexity score to CSV
                score = scores[i].item()
                with open(args.output_csv, 'a') as f:
                    f.write(f"{os.path.basename(img_path)},{score:.7f}\n")
                
                # Get original dimensions
                width, height = original_sizes[i]
                
                # Process complexity map
                ic_map = maps[i:i+1]
                ic_map = F.interpolate(ic_map, (height, width), mode='bilinear')
                ic_map_np = ic_map.squeeze().cpu().numpy()
                
                # Save map if enabled
                if args.save_maps:
                    map_path = os.path.join(vis_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_map.npy")
                    np.save(map_path, ic_map_np)
                
                # Save visualization if enabled
                if args.save_visualizations:
                    with Image.open(img_path) as img:
                        img_array = np.array(img.convert("RGB"))
                        vis_img = blend(img_array, (ic_map_np * 255).astype(np.uint8))
                        vis_path = os.path.join(vis_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_vis.png")
                        cv2.imwrite(vis_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
                
                processed += 1
            except Exception as e:
                logging.error(f"Error processing {img_path}: {str(e)}")
                skipped += 1
        
        return skipped, processed
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            logging.warning(f"OOM with batch size {args.batch_size}")
            return len(image_paths), 0  # Skip entire batch
        logging.error(f"Runtime error: {str(e)}")
        if not args.skip_errors:
            raise
        return len(image_paths), 0
    except Exception as e:
        logging.error(f"Batch processing failed: {str(e)}")
        if not args.skip_errors:
            raise
        return len(image_paths), 0

def process_directory(img_dir, model, device, transform, args):
    """Process all images in a directory"""
    # Find all image files recursively
    img_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
    img_files = [
        os.path.join(root, f) 
        for root, _, files in os.walk(img_dir) 
        for f in files 
        if f.lower().endswith(img_exts)
    ]
    
    if not img_files:
        logging.warning(f"No valid images found in {img_dir}")
        return
    
    # Create visualization directory if needed
    vis_dir = ""
    if args.save_maps or args.save_visualizations:
        output_dir = os.path.dirname(args.output_csv) or "."
        vis_dir = os.path.join(output_dir, "visualization")
        os.makedirs(vis_dir, exist_ok=True)
        logging.info(f"Saving visual outputs to: {vis_dir}")
    
    # Initialize CSV file
    with open(args.output_csv, 'w') as f:
        f.write("image,complexity_score\n")
    
    # Process images in batches
    total = len(img_files)
    processed_count = skipped_count = 0
    batch_size = args.batch_size
    
    logging.info(f"Processing {total} images in batches of {batch_size}")
    progress = tqdm(total=total, desc="Processing Images", unit="img")
    start_time = time.time()
    
    i = 0
    while i < total:
        batch = img_files[i:i + batch_size]
        skipped, processed = process_batch(batch, model, device, transform, args, vis_dir)
        skipped_count += skipped
        processed_count += processed
        
        # Adjust batch size on OOM
        if processed == 0 and batch_size > 1:
            batch_size = max(1, batch_size // 2)
            logging.info(f"Reducing batch size to {batch_size}")
            continue
        
        i += len(batch)
        progress.update(len(batch))
        
        # Increase batch size if possible
        if batch_size < args.batch_size and processed > 0:
            batch_size = min(args.batch_size, batch_size * 2)
    
    progress.close()
    
    # Print performance summary
    elapsed = time.time() - start_time
    logging.info(f"Processed {processed_count} images in {elapsed:.1f} seconds")
    logging.info(f"Images/sec: {processed_count/elapsed:.1f}")
    logging.info(f"Skipped: {skipped_count} images")

def main():
    """Main execution function"""
    args = parse_arguments()
    logger = setup_logger(args.log_file)
    
    # Create output directory for CSV
    output_dir = os.path.dirname(args.output_csv) or "."
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device (GPU/CPU)
    device = torch.device(
        f"cuda:{args.device}" 
        if args.device >= 0 and torch.cuda.is_available() 
        else "cpu"
    )
    logging.info(f"Using device: {device}")
    
    try:
        # Load model
        logging.info(f"Loading model from {args.model}")
        model = ICNet().to(device)
        
        # Debug model structure
        logging.debug("Model architecture:")
        logging.debug(model)
        
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.eval()
        
        # Image transformation pipeline
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Process input based on type
        if os.path.isfile(args.input):
            logging.info(f"Processing single image: {args.input}")
            # Setup visualization directory if needed
            vis_dir = ""
            if args.save_maps or args.save_visualizations:
                vis_dir = os.path.join(output_dir, "visualization")
                os.makedirs(vis_dir, exist_ok=True)
            
            # Initialize CSV
            with open(args.output_csv, 'w') as f:
                f.write("image,complexity_score\n")
            
            # Process single image
            skipped, processed = process_batch(
                [args.input], model, device, transform, args, vis_dir
            )
            if not processed:
                logging.error(f"Failed to process {args.input}")
        elif os.path.isdir(args.input):
            logging.info(f"Processing directory: {args.input}")
            process_directory(args.input, model, device, transform, args)
        else:
            raise ValueError(f"Invalid input: {args.input}")
        
        logging.info("Processing completed successfully")
    except Exception as e:
        logging.critical(f"Fatal error: {str(e)}")
        logging.debug(traceback.format_exc())
        exit(1)

if __name__ == "__main__":
    main()