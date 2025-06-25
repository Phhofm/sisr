#!/usr/bin/env python3
"""
DPID Image Downscaler
Created by Philip Hofmann with assistance from DeepSeek AI

Purpose:
  Batch downscale images using the DPID algorithm with λ=1.0 for maximum detail retention.
  Processes entire directories while preserving folder structures.

Key Features:
  - Utilizes DPID (Differentiable Programming for Image Downscaling) algorithm with λ=1.0
  - Maintains optimal detail with (scale-1)/scale ratio
  - Creates necessary output directories automatically
  - Parallel processing support
  - Recursive directory traversal
  - Flexible output formats
  - Comprehensive error handling

Usage:
  dpid_downscaler.py [input_folder] [scale] [output_folder] [options]

Dependencies:
  pepedpid (DPID implementation by umzi): https://github.com/umzi/pepedpid
  pepeline (I/O library)

Example:
  dpid_downscaler.py ~/photos 3 ./downscaled -r -t 8 --output-ext .webp
"""

import os
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pepeline import read, save, ImgFormat
from pepedpid import dpid_resize

# Supported image formats (extend as needed)
SUPPORTED_INPUT_EXTS = {'.webp', '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.tif', '.jfif'}
SUPPORTED_OUTPUT_EXTS = {'.png', '.jpg', '.jpeg', '.webp'}

def process_image(input_path, output_path, scale, verbose=False):
    """Process a single image with DPID resizing using λ=1.0
    
    Args:
        input_path: Source image path
        output_path: Target output path
        scale: Downscaling factor (integer)
        verbose: Enable progress output
    Returns:
        True on success, False on failure
    """
    try:
        # Read image in float32 format for maximum precision
        img = read(input_path, format=ImgFormat.F32)
        h, w = img.shape[:2]
        
        # Compute DPID factor (λ=1.0)
        factor = (scale - 1) / scale
        
        # Calculate target dimensions (1/scale of original)
        target_h = max(1, int(round(h / scale)))
        target_w = max(1, int(round(w / scale)))
        
        # Apply DPID resizing with λ=1.0 configuration
        resized = dpid_resize(img, target_h, target_w, factor)
        
        # Save processed image
        save(resized, output_path)
        
        if verbose:
            print(f"Processed: {os.path.basename(input_path)} -> {os.path.basename(output_path)} [Size: {target_w}x{target_h}]")
        return True
    except Exception as e:
        print(f"Error processing {os.path.basename(input_path)}: {str(e)}", file=sys.stderr)
        return False
    
def main():
    # Initialize argument parser with description and help formatting
    parser = argparse.ArgumentParser(
        description='DPID Image Downscaler (λ=1.0) - Detail-Preserving Image Resizing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('input_folder', help='Path to input folder containing images')
    parser.add_argument('scale', type=int, help='Downscale factor (integer ≥ 2)')
    parser.add_argument('output_folder', help='Output directory for processed images')
    
    # Optional arguments
    parser.add_argument('--output-ext', default='.png', 
                        help=f'Output format ({", ".join(SUPPORTED_OUTPUT_EXTS)})')
    parser.add_argument('--recursive', '-r', action='store_true', 
                        help='Process subdirectories recursively')
    parser.add_argument('--threads', '-t', type=int, default=4, 
                        help='Parallel processing threads (0=disable)')
    parser.add_argument('--skip-existing', action='store_true', 
                        help='Skip processing if output file exists')
    parser.add_argument('--verbose', '-v', action='store_true', 
                        help='Show detailed processing information')
    parser.add_argument('--dry-run', action='store_true', 
                        help='Preview operations without actual processing')
    
    args = parser.parse_args()

    # --- Input Validation ---
    if args.scale < 2:
        raise ValueError("Scale factor must be ≥ 2")
    if args.output_ext.lower() not in SUPPORTED_OUTPUT_EXTS:
        raise ValueError(f"Unsupported output extension. Valid: {', '.join(SUPPORTED_OUTPUT_EXTS)}")
    if not os.path.exists(args.input_folder):
        raise NotADirectoryError(f"Input folder not found: {args.input_folder}")

    # --- Create Output Directory ---
    # Create the main output directory if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Check for input/output collision after creating output directory
    if os.path.samefile(args.input_folder, args.output_folder):
        raise ValueError("Input and output folders must be different")

    # --- Configuration ---
    # Factor is now calculated per-image in process_image()

    # --- Collect Image Paths ---
    image_paths = []
    if args.recursive:
        # Recursive directory traversal
        for root, _, files in os.walk(args.input_folder):
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext in SUPPORTED_INPUT_EXTS:
                    input_path = os.path.join(root, filename)
                    # Preserve relative path structure
                    rel_path = os.path.relpath(input_path, args.input_folder)
                    output_path = os.path.join(
                        args.output_folder, 
                        os.path.splitext(rel_path)[0] + args.output_ext
                    )
                    image_paths.append((input_path, output_path))
    else:
        # Only process top-level directory
        for filename in os.listdir(args.input_folder):
            filepath = os.path.join(args.input_folder, filename)
            if os.path.isfile(filepath):
                ext = os.path.splitext(filename)[1].lower()
                if ext in SUPPORTED_INPUT_EXTS:
                    output_path = os.path.join(
                        args.output_folder, 
                        os.path.splitext(filename)[0] + args.output_ext
                    )
                    image_paths.append((filepath, output_path))

    if not image_paths:
        print("No valid images found in input directory")
        return

   # --- Processing Setup ---
    processed_count = 0
    skipped_count = 0

    if args.dry_run:
        print(f"Dry run: Would process {len(image_paths)} images")
        print(f"Output directory: {args.output_folder}")
        print(f"Scale factor: 1/{args.scale} (dimensions calculated per-image)")
        return  # FIX: Removed dimension display

    def process_task(input_path, output_path):
        """Wrapper function for processing individual images"""
        nonlocal skipped_count
        
        # Skip existing files if requested
        if args.skip_existing and os.path.exists(output_path):
            skipped_count += 1
            if args.verbose:
                print(f"Skipping existing: {os.path.basename(output_path)}")
            return False
            
        # Create output subdirectories if needed
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        # Process image and return result
        return process_image(input_path, output_path, args.scale, args.verbose)

    # --- Parallel Processing ---
    if args.threads > 0:
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = [executor.submit(process_task, ip, op) for ip, op in image_paths]
            for future in as_completed(futures):
                if future.result():
                    processed_count += 1
    # --- Serial Processing ---
    else:
        for input_path, output_path in image_paths:
            if process_task(input_path, output_path):
                processed_count += 1

    # --- Results Summary ---
    print(f"\n{'Operation complete:':<20}")
    print(f"{'Processed:':<12} \033[1m{processed_count}\033[0m images")
    print(f"{'Skipped:':<12} \033[1m{skipped_count}\033[0m files")
    print(f"{'Failed:':<12} \033[1m{len(image_paths) - processed_count - skipped_count}\033[0m files")
    print(f"{'Output:':<12} \033[1m{args.output_folder}\033[0m")
    print(f"{'Scale factor:':<12} \033[1m1/{args.scale}\033[0m")  # FIX: Added scale display

if __name__ == "__main__":
    main()