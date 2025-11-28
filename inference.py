"""
Inference script to generate captions for new images
"""

import torch
from PIL import Image
from torchvision import transforms
import argparse
import os

from models import ImageCaptioningModel
from vocabulary import Vocabulary
from config import *


def load_image(image_path, transform=None):
    """
    Load and preprocess a single image
    
    Args:
        image_path: Path to image file
        transform: Torchvision transforms
        
    Returns:
        Preprocessed image tensor [1, 3, 224, 224]
    """
    image = Image.open(image_path).convert('RGB')
    
    if transform:
        image = transform(image)
    
    # Add batch dimension
    return image.unsqueeze(0)


def generate_caption(model, image, vocab, device, max_length=30):
    """
    Generate caption for a single image
    
    Args:
        model: Trained image captioning model
        image: Preprocessed image tensor
        vocab: Vocabulary object
        device: Device
        max_length: Maximum caption length
        
    Returns:
        Generated caption string
    """
    model.eval()
    
    with torch.no_grad():
        image = image.to(device)
        caption_ids = model.generate_caption(image, max_length)
        caption = vocab.decode_caption(caption_ids[0], skip_special_tokens=True)
    
    return caption


def main(args):
    """Main inference function"""
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"🖼️  IMAGE CAPTIONING INFERENCE")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Image: {args.image_path}")
    print(f"{'='*70}\n")
    
    # Load vocabulary
    print("📚 Loading vocabulary...")
    if not os.path.exists(args.vocab_path):
        print(f"❌ Error: Vocabulary not found at {args.vocab_path}")
        print(f"   Please train the model first!")
        return
    
    vocab = Vocabulary. load(args.vocab_path)
    
    # Load model
    print("🏗️  Loading model...")
    if not os.path.exists(args. model_path):
        print(f"❌ Error: Model checkpoint not found at {args.model_path}")
        print(f"   Please train the model first!")
        return
    
    model = ImageCaptioningModel(
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        vocab_size=len(vocab),
        num_layers=NUM_LAYERS,
        dropout=0.0,  # No dropout for inference
        freeze_encoder=False
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Model loaded from {args.model_path}")
    print(f"   Trained for {checkpoint. get('epoch', '?')} epochs")
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms. Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])
    
    # Load and process image
    print("\n📷 Processing image...")
    if not os.path.exists(args. image_path):
        print(f"❌ Error: Image not found at {args.image_path}")
        return
    
    image = load_image(args.image_path, transform)
    
    # Generate caption
    print("✨ Generating caption.. .\n")
    caption = generate_caption(model, image, vocab, device, MAX_CAPTION_LENGTH)
    
    # Display result
    print(f"{'='*70}")
    print(f"📝 GENERATED CAPTION")
    print(f"{'='*70}")
    print(f"{caption}")
    print(f"{'='*70}\n")
    
    # Save to file if specified
    if args.output:
        with open(args.output, 'w') as f:
            f.write(caption)
        print(f"💾 Caption saved to {args.output}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate caption for an image')
    parser. add_argument('--image_path', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--vocab_path', type=str, default='checkpoints/vocab.pkl',
                       help='Path to vocabulary file')
    parser.add_argument('--output', type=str, default='',
                       help='Path to save generated caption (optional)')
    
    args = parser.parse_args()
    main(args)