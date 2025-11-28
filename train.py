"""
Training script for Image Captioning - Scenario B
Fine-tuned encoder + LSTM decoder with early stopping
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import argparse
import time

from config import *
from models import ImageCaptioningModel
from dataset import get_dataloaders
from vocabulary import Vocabulary
from utils import save_checkpoint, load_checkpoint, EarlyStopping, plot_training_curves, count_parameters
import pandas as pd


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Train for one epoch
    
    Args:
        model: Image captioning model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        
    Returns:
        Average training loss
    """
    model. train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{MAX_EPOCHS} [Train]')
    
    for batch_idx, (images, captions) in enumerate(pbar):
        images = images.to(device)
        captions = captions.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images, captions)
        
        # Calculate loss (ignore padding)
        # outputs: [batch_size, max_length, vocab_size]
        # captions: [batch_size, max_length]
        loss = criterion(outputs. view(-1, outputs.size(-1)), captions.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        
        # Update weights
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss. item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/(batch_idx+1):. 4f}'
        })
        
        # Print periodically
        if (batch_idx + 1) % PRINT_FREQ == 0:
            print(f'  Batch [{batch_idx+1}/{len(dataloader)}] Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, criterion, device, epoch):
    """
    Validate model
    
    Args:
        model: Image captioning model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
        
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{MAX_EPOCHS} [Val]  ')
        
        for images, captions in pbar:
            images = images.to(device)
            captions = captions.to(device)
            
            # Forward pass
            outputs = model(images, captions)
            
            # Calculate loss
            loss = criterion(outputs.view(-1, outputs.size(-1)), captions.view(-1))
            
            total_loss += loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main(args):
    """Main training function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"🚀 SCENARIO B: Fine-tuned Encoder + LSTM Decoder")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"{'='*70}\n")
    
    # Create directories
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Check if dataset exists
    if not os.path.exists(args.captions_file):
        print(f"❌ Error: Captions file not found at {args.captions_file}")
        print(f"   Please download Flickr8k dataset first!")
        return
    
    # Build vocabulary
    print("📚 Building vocabulary...")
    df = pd.read_csv(args.captions_file)
    vocab = Vocabulary(min_freq=MIN_WORD_FREQ)
    vocab.build_vocab(df['caption']. tolist())
    vocab.save(os.path.join(SAVE_DIR, 'vocab.pkl'))
    
    # Get dataloaders
    print("\n📂 Loading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(
        args.image_dir,
        args.captions_file,
        vocab,
        batch_size=BATCH_SIZE,
        num_workers=args.num_workers
    )
    
    # Create model
    print("\n🏗️  Creating model...")
    model = ImageCaptioningModel(
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        vocab_size=len(vocab),
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        freeze_encoder=FREEZE_ENCODER_INITIALLY
    ).to(device)
    
    total_params, trainable_params = count_parameters(model)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen parameters: {total_params - trainable_params:,}")
    
    # Loss function (ignore padding token)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab('<pad>'))
    
    # Optimizer with separate learning rates (KEY for Scenario B)
    print(f"\n⚙️  Optimizer: AdamW")
    print(f"   Decoder LR: {LEARNING_RATE_DECODER}")
    print(f"   Encoder LR: {LEARNING_RATE_ENCODER}")
    print(f"   Weight Decay: {WEIGHT_DECAY}")
    
    optimizer = optim.AdamW([
        {'params': model.decoder.parameters(), 'lr': LEARNING_RATE_DECODER},
        {'params': model.encoder.parameters(), 'lr': LEARNING_RATE_ENCODER}
    ], weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, verbose=True)
    
    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if args.resume:
        if os.path.exists(args.resume):
            print(f"\n📥 Resuming from checkpoint: {args.resume}")
            start_epoch, _ = load_checkpoint(args.resume, model, optimizer)
        else:
            print(f"⚠️  Checkpoint not found: {args.resume}, starting from scratch")
    
    # Training loop
    print(f"\n{'='*70}")
    print(f"🎯 Training Configuration")
    print(f"{'='*70}")
    print(f"Max Epochs: {MAX_EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Unfreeze Encoder After Epoch: {UNFREEZE_AFTER_EPOCH}")
    print(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
    print(f"{'='*70}\n")
    
    training_start_time = time.time()
    
    for epoch in range(start_epoch, MAX_EPOCHS):
        epoch_start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"📅 Epoch {epoch + 1}/{MAX_EPOCHS}")
        print(f"{'='*70}")
        
        # Unfreeze encoder layers after specified epoch (SCENARIO B KEY STEP)
        if epoch == UNFREEZE_AFTER_EPOCH and FREEZE_ENCODER_INITIALLY:
            print(f"\n🔓 UNFREEZING ENCODER LAYERS: {ENCODER_LAYERS_TO_FINETUNE}")
            model.encoder.unfreeze_layers(ENCODER_LAYERS_TO_FINETUNE)
            
            total_params, trainable_params = count_parameters(model)
            print(f"   Trainable parameters now: {trainable_params:,}")
            print(f"   This allows fine-tuning for better image features!  🎨\n")
        
        # Train one epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch + 1)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device, epoch + 1)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"\n📊 Epoch {epoch + 1} Summary:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss:   {val_loss:.4f}")
        print(f"   Time: {epoch_time:.2f}s")
        
        # Check for overfitting/underfitting
        if epoch > 0:
            if val_loss > train_loss * 1.5:
                print(f"   ⚠️  WARNING: Possible overfitting (val_loss >> train_loss)")
            elif train_loss > 3.0 and val_loss > 3.0:
                print(f"   ⚠️  WARNING: Possible underfitting (both losses high)")
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"   ✨ New best model!  Val Loss: {val_loss:.4f}")
        
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'vocab': vocab,
            'config': {
                'embed_size': EMBED_SIZE,
                'hidden_size': HIDDEN_SIZE,
                'num_layers': NUM_LAYERS,
                'dropout': DROPOUT
            }
        }, is_best, filename=os.path.join(SAVE_DIR, f'checkpoint_epoch_{epoch + 1}.pth'))
        
        # Early stopping check
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"\n🛑 Early stopping triggered at epoch {epoch + 1}")
            print(f"   Validation loss hasn't improved for {EARLY_STOPPING_PATIENCE} epochs")
            break
        
        # Plot training curves every 2 epochs
        if (epoch + 1) % 2 == 0:
            plot_training_curves(train_losses, val_losses, 
                               save_path=os.path.join(LOG_DIR, 'training_curves.png'))
    
    # Training completed
    total_time = time.time() - training_start_time
    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETED!")
    print(f"{'='*70}")
    print(f"Total Training Time: {total_time/3600:.2f} hours")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Model: {os.path.join(SAVE_DIR, 'best_model.pth')}")
    print(f"Training Curves: {os.path.join(LOG_DIR, 'training_curves.png')}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Image Captioning Model - Scenario B')
    parser. add_argument('--image_dir', type=str, default=DEFAULT_IMAGE_DIR,
                       help='Path to image directory')
    parser. add_argument('--captions_file', type=str, default=DEFAULT_CAPTIONS_FILE,
                       help='Path to captions CSV file')
    parser.add_argument('--resume', type=str, default='',
                       help='Path to checkpoint to resume from')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data loading workers')
    
    args = parser.parse_args()
    main(args)