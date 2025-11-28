"""
Model architectures for image captioning
Encoder: Pretrained ResNet50 with fine-tuning
Decoder: LSTM with optional attention
"""

import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    """
    CNN Encoder using pretrained ResNet50
    Supports gradual unfreezing for fine-tuning (Scenario B)
    """
    def __init__(self, embed_size, freeze=True):
        """
        Args:
            embed_size: Size of embedding vector
            freeze: Whether to freeze encoder initially
        """
        super(EncoderCNN, self).__init__()
        
        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=True)
        
        # Remove final FC layer (we'll add our own)
        modules = list(resnet.children())[:-1]  # All except FC
        self.resnet = nn.Sequential(*modules)
        
        # Add custom embedding layer
        self.embed = nn.Linear(resnet. fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)
        
        # Freeze all layers initially (for Scenario B)
        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False
            print("🔒 Encoder frozen initially (Scenario B)")
        
    def forward(self, images):
        """
        Forward pass
        
        Args:
            images: Batch of images [batch_size, 3, 224, 224]
            
        Returns:
            features: Encoded features [batch_size, embed_size]
        """
        # Extract features
        with torch.set_grad_enabled(self.training):
            features = self.resnet(images)
        
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.bn(features)
        
        return features
    
    def unfreeze_layers(self, layer_names=['layer4']):
        """
        Unfreeze specific layers for fine-tuning
        
        Args:
            layer_names: List of layer names to unfreeze (e.g., ['layer4'])
        """
        for name, child in self.resnet.named_children():
            if name in layer_names:
                print(f"🔓 Unfreezing encoder layer: {name}")
                for param in child.parameters():
                    param.requires_grad = True


class DecoderLSTM(nn.Module):
    """
    LSTM Decoder with teacher forcing
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.3):
        """
        Args:
            embed_size: Size of word embeddings
            hidden_size: Size of LSTM hidden state
            vocab_size: Size of vocabulary
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(DecoderLSTM, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # Word embedding layer
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        """
        Forward pass with teacher forcing
        
        Args:
            features: Image features from encoder [batch_size, embed_size]
            captions: Ground truth captions [batch_size, max_length]
            
        Returns:
            outputs: Predicted word scores [batch_size, max_length, vocab_size]
        """
        # Embed captions
        embeddings = self.embed(captions)  # [batch_size, max_length, embed_size]
        
        # Concatenate image features with caption embeddings
        # features: [batch_size, embed_size] -> [batch_size, 1, embed_size]
        # We prepend image features and remove last word (teacher forcing)
        embeddings = torch.cat(
            (features. unsqueeze(1), embeddings[:, :-1, :]),
            dim=1
        )
        
        # Pass through LSTM
        hiddens, _ = self.lstm(embeddings)  # [batch_size, max_length, hidden_size]
        
        # Apply dropout
        hiddens = self. dropout(hiddens)
        
        # Generate word scores
        outputs = self.linear(hiddens)  # [batch_size, max_length, vocab_size]
        
        return outputs
    
    def generate(self, features, max_length=30, start_token=1, end_token=2):
        """
        Generate caption using greedy search
        
        Args:
            features: Image features [batch_size, embed_size]
            max_length: Maximum caption length
            start_token: Index of <start> token
            end_token: Index of <end> token
            
        Returns:
            captions: Generated captions [batch_size, max_length]
        """
        batch_size = features.size(0)
        captions = []
        
        # Start with image features
        inputs = features. unsqueeze(1)  # [batch_size, 1, embed_size]
        states = None
        
        for i in range(max_length):
            # Forward pass through LSTM
            hiddens, states = self.lstm(inputs, states)
            
            # Get word scores
            outputs = self.linear(hiddens. squeeze(1))  # [batch_size, vocab_size]
            
            # Get predicted word
            predicted = outputs.argmax(1)  # [batch_size]
            captions.append(predicted)
            
            # Prepare input for next timestep
            inputs = self.embed(predicted). unsqueeze(1)  # [batch_size, 1, embed_size]
        
        # Stack predictions
        captions = torch.stack(captions, 1)  # [batch_size, max_length]
        
        return captions


class ImageCaptioningModel(nn. Module):
    """
    Complete image captioning model (Encoder + Decoder)
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.3, freeze_encoder=True):
        """
        Args:
            embed_size: Size of embeddings
            hidden_size: Size of LSTM hidden state
            vocab_size: Size of vocabulary
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            freeze_encoder: Whether to freeze encoder initially
        """
        super(ImageCaptioningModel, self).__init__()
        
        self.encoder = EncoderCNN(embed_size, freeze=freeze_encoder)
        self.decoder = DecoderLSTM(embed_size, hidden_size, vocab_size, num_layers, dropout)
        
    def forward(self, images, captions):
        """
        Forward pass
        
        Args:
            images: Batch of images
            captions: Batch of captions
            
        Returns:
            outputs: Predicted word scores
        """
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
    
    def generate_caption(self, images, max_length=30, start_token=1, end_token=2):
        """
        Generate captions for images
        
        Args:
            images: Batch of images
            max_length: Maximum caption length
            
        Returns:
            captions: Generated captions
        """
        with torch.no_grad():
            features = self.encoder(images)
            captions = self.decoder.generate(features, max_length, start_token, end_token)
        return captions