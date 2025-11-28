"""
Configuration file for Image Captioning - Scenario B
Fine-tuned Encoder (ResNet50) + LSTM Decoder
"""

# ============================================================================
# SCENARIO B CONFIGURATION
# ============================================================================
SCENARIO = "B"  # Fine-tuned encoder + LSTM decoder

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
ENCODER_TYPE = "resnet50"  # Pretrained ResNet50
DECODER_TYPE = "lstm"
EMBED_SIZE = 300
HIDDEN_SIZE = 512
NUM_LAYERS = 1
ATTENTION = False  # Can be enabled for better performance

# ============================================================================
# TRAINING HYPERPARAMETERS (SCENARIO B SPECIFIC)
# ============================================================================
BATCH_SIZE = 32
MAX_EPOCHS = 20  # Scenario B: typically 8-20 epochs
LEARNING_RATE_DECODER = 1e-4  # Higher LR for decoder
LEARNING_RATE_ENCODER = 1e-5  # Lower LR for fine-tuning encoder
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 5.0
DROPOUT = 0.3

# ============================================================================
# ENCODER FINE-TUNING STRATEGY (KEY FOR SCENARIO B)
# ============================================================================
FREEZE_ENCODER_INITIALLY = True  # Start with frozen encoder
UNFREEZE_AFTER_EPOCH = 5  # Unfreeze after decoder converges
ENCODER_LAYERS_TO_FINETUNE = ['layer4']  # Only fine-tune last ResNet block

# ============================================================================
# EARLY STOPPING & REGULARIZATION
# ============================================================================
EARLY_STOPPING_PATIENCE = 5
USE_CIDER_FOR_EARLY_STOP = True
LABEL_SMOOTHING = 0.0  # Optional: 0.1 for transformer-like training

# ============================================================================
# DATA SPLIT (Flickr8k: 8,000 images total)
# ============================================================================
TRAIN_SIZE = 6000  # 75%
VAL_SIZE = 1000    # 12. 5%
TEST_SIZE = 1000   # 12.5%
RANDOM_SEED = 42

# ============================================================================
# VOCABULARY SETTINGS
# ============================================================================
MIN_WORD_FREQ = 5  # Filter rare words
MAX_CAPTION_LENGTH = 30

# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================
IMAGE_SIZE = 256  # Resize short side to this
CROP_SIZE = 224   # Center/random crop to this (ResNet50 input)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # ImageNet statistics
NORMALIZE_STD = [0.229, 0.224, 0.225]

# ============================================================================
# DATA AUGMENTATION (TRAIN ONLY)
# ============================================================================
RANDOM_HORIZONTAL_FLIP = 0.5
COLOR_JITTER_BRIGHTNESS = 0.2
COLOR_JITTER_CONTRAST = 0.2
COLOR_JITTER_SATURATION = 0.2
COLOR_JITTER_HUE = 0.0

# ============================================================================
# SPECIAL TOKENS
# ============================================================================
PAD_TOKEN = '<pad>'
START_TOKEN = '<start>'
END_TOKEN = '<end>'
UNK_TOKEN = '<unk>'

# ============================================================================
# CHECKPOINTING & SAVING
# ============================================================================
SAVE_DIR = 'checkpoints'
SAVE_EVERY = 1  # Save every epoch
AUTO_SAVE_BEFORE_EXPIRY = True  # For Colab session protection

# ============================================================================
# EVALUATION
# ============================================================================
EVAL_EVERY = 1  # Evaluate every epoch
BEAM_SIZE = 5  # For beam search during inference
NUM_SAMPLE_CAPTIONS = 5  # Number of samples to print during evaluation

# ============================================================================
# LOGGING
# ============================================================================
LOG_DIR = 'logs'
PRINT_FREQ = 100  # Print every N batches

# ============================================================================
# DATASET PATHS (will be overridden by command line args)
# ============================================================================
DEFAULT_IMAGE_DIR = 'data/flickr8k/Images'
DEFAULT_CAPTIONS_FILE = 'data/flickr8k/captions.txt'