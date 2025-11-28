"""
Dataset loader for Flickr8k with train/val/test split and augmentation
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import numpy as np
from config import *

class Flickr8kDataset(Dataset):
    """
    Flickr8k dataset loader
    Each image has ~5 captions - we use all for training
    """
    def __init__(self, image_dir, captions_df, vocab, transform=None, max_length=30):
        """
        Args:
            image_dir: Directory containing images
            captions_df: DataFrame with 'image' and 'caption' columns
            vocab: Vocabulary object
            transform: Image transforms
            max_length: Maximum caption length
        """
        self. image_dir = image_dir
        self.captions_df = captions_df. reset_index(drop=True)
        self.vocab = vocab
        self.transform = transform
        self.max_length = max_length
        
    def __len__(self):
        return len(self.captions_df)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Preprocessed image tensor
            caption: Encoded caption tensor
        """
        # Get image filename and caption
        img_name = self.captions_df.iloc[idx]['image']
        caption = self.captions_df.iloc[idx]['caption']
        
        # Load image
        img_path = os.path.join(self.image_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️ Error loading image {img_path}: {e}")
            # Return a blank image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Encode caption
        caption_encoded = self.vocab.encode_caption(caption, self.max_length)
        
        return image, torch.tensor(caption_encoded, dtype=torch.long)


def get_transforms(train=True):
    """
    Get image transforms with augmentation for training
    
    Args:
        train: Whether to apply training augmentation
        
    Returns:
        torchvision.transforms.Compose object
    """
    if train:
        return transforms.Compose([
            transforms. Resize(IMAGE_SIZE),
            transforms.RandomCrop(CROP_SIZE),
            transforms. RandomHorizontalFlip(p=RANDOM_HORIZONTAL_FLIP),
            transforms.ColorJitter(
                brightness=COLOR_JITTER_BRIGHTNESS,
                contrast=COLOR_JITTER_CONTRAST,
                saturation=COLOR_JITTER_SATURATION,
                hue=COLOR_JITTER_HUE
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        ])
    else:
        # No augmentation for validation/test
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms. CenterCrop(CROP_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        ])


def split_dataset(captions_file, train_size=6000, val_size=1000, test_size=1000, seed=42):
    """
    Split dataset by IMAGES (not captions) to prevent data leakage
    
    Args:
        captions_file: Path to captions CSV file
        train_size: Number of images for training
        val_size: Number of images for validation
        test_size: Number of images for testing
        seed: Random seed
        
    Returns:
        train_df, val_df, test_df: DataFrames for each split
    """
    print("Splitting dataset...")
    
    # Load captions
    df = pd.read_csv(captions_file)
    print(f"Total captions: {len(df)}")
    
    # Get unique images
    unique_images = df['image'].unique()
    print(f"Total unique images: {len(unique_images)}")
    
    # Shuffle images
    np.random.seed(seed)
    np.random. shuffle(unique_images)
    
    # Split images
    train_images = unique_images[:train_size]
    val_images = unique_images[train_size:train_size + val_size]
    test_images = unique_images[train_size + val_size:train_size + val_size + test_size]
    
    # Create split dataframes (keep all captions for each image)
    train_df = df[df['image'].isin(train_images)]
    val_df = df[df['image']. isin(val_images)]
    test_df = df[df['image'].isin(test_images)]
    
    print(f"✅ Dataset split complete:")
    print(f"   Train: {len(train_images)} images, {len(train_df)} captions")
    print(f"   Val:   {len(val_images)} images, {len(val_df)} captions")
    print(f"   Test:  {len(test_images)} images, {len(test_df)} captions")
    
    return train_df, val_df, test_df


def collate_fn(batch):
    """
    Custom collate function for DataLoader
    
    Args:
        batch: List of (image, caption) tuples
        
    Returns:
        images: Batched image tensor
        captions: Batched caption tensor
    """
    images, captions = zip(*batch)
    images = torch.stack(images, 0)
    captions = torch.stack(captions, 0)
    return images, captions


def get_dataloaders(image_dir, captions_file, vocab, batch_size=32, num_workers=2):
    """
    Create train, val, test dataloaders
    
    Args:
        image_dir: Directory containing images
        captions_file: Path to captions file
        vocab: Vocabulary object
        batch_size: Batch size
        num_workers: Number of data loading workers
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Split dataset
    train_df, val_df, test_df = split_dataset(
        captions_file, 
        train_size=TRAIN_SIZE,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        seed=RANDOM_SEED
    )
    
    # Save split files for reproducibility
    os.makedirs('data/splits', exist_ok=True)
    train_df.to_csv('data/splits/train.csv', index=False)
    val_df.to_csv('data/splits/val.csv', index=False)
    test_df.to_csv('data/splits/test.csv', index=False)
    print("✅ Split files saved to data/splits/")
    
    # Create datasets
    train_dataset = Flickr8kDataset(
        image_dir, train_df, vocab,
        transform=get_transforms(train=True),
        max_length=MAX_CAPTION_LENGTH
    )
    
    val_dataset = Flickr8kDataset(
        image_dir, val_df, vocab,
        transform=get_transforms(train=False),
        max_length=MAX_CAPTION_LENGTH
    )
    
    test_dataset = Flickr8kDataset(
        image_dir, test_df, vocab,
        transform=get_transforms(train=False),
        max_length=MAX_CAPTION_LENGTH
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader