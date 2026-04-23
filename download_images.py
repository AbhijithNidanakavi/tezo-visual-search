"""
Image Downloader for Photo Dataset
--------------------------------
This script downloads and optimizes images from URLs in a photos.csv file.

Requirements:
    pip install pandas pillow requests tqdm

Usage:
    1. Ensure photos_url.csv is in the same directory as this script
    2. Run the script: python download_images.py
    3. Images will be downloaded to the 'data/images' folder

Note: 
    - Default image size is 800x800 pixels (maintains aspect ratio)
    - Images are saved as optimized JPEGs
    - You can modify num_images parameter to download fewer images
    - Approximate size of the dataset is 1.5GB and total images are 25,000 images
"""

import pandas as pd
import requests
import os
from PIL import Image
from io import BytesIO
from tqdm import tqdm

def download_images(num_images=None, output_dir="data/images", target_size=(800, 800), csv_path="photos_url.csv"):
    """
    Download and optimize images from photos.csv
    
    Args:
        num_images: Number of images to download (default: all images in CSV)
        output_dir: Directory to save images (default: 'data/images')
        target_size: Max image dimensions (default: (800, 800))
        csv_path: CSV file containing image URLs (default: 'photos_url.csv')
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read CSV and prepare dataset
    df = pd.read_csv(csv_path)
    if num_images:
        df = df.head(num_images)
    
    # Download images
    print(f"Downloading {len(df)} images to {output_dir}...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # Download and save image
            filename = f"{(idx+1):04d}.jpg"
            output_path = os.path.join(output_dir, filename)
            
            response = requests.get(row['photo_image_url'], timeout=10)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                img.thumbnail(target_size, Image.Resampling.LANCZOS)
                img.save(output_path, 'JPEG', quality=85, optimize=True)
                
        except Exception as e:
            print(f"Error downloading image {idx+1}")
            continue

if __name__ == "__main__":
    # Download all images (or modify num_images to download fewer)
    download_images(num_images=None)
