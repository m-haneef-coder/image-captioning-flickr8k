"""
Evaluation metrics for image captioning
BLEU, METEOR, CIDEr, ROUGE-L
"""

import torch
from tqdm import tqdm
import numpy as np


def evaluate_model(model, dataloader, vocab, device, max_length=30):
    """
    Evaluate model and generate captions for all samples
    
    Args:
        model: Image captioning model
        dataloader: DataLoader for evaluation
        vocab: Vocabulary object
        device: Device to evaluate on
        max_length: Maximum caption length
        
    Returns:
        references: List of reference captions
        hypotheses: List of generated captions
    """
    model.eval()
    
    references = []
    hypotheses = []
    
    with torch.no_grad():
        for images, captions in tqdm(dataloader, desc='Generating captions'):
            images = images.to(device)
            
            # Generate captions
            generated = model.generate_caption(images, max_length)
            
            # Convert to text
            for i in range(images.size(0)):
                # Reference caption
                ref_caption = vocab. decode_caption(captions[i], skip_special_tokens=True)
                references.append([ref_caption])  # Wrapped in list for BLEU
                
                # Generated caption
                gen_caption = vocab.decode_caption(generated[i], skip_special_tokens=True)
                hypotheses.append(gen_caption)
    
    return references, hypotheses


def calculate_bleu_scores(references, hypotheses):
    """
    Calculate BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
    
    Args:
        references: List of reference captions (each a list of strings)
        hypotheses: List of generated captions (strings)
        
    Returns:
        Dictionary with BLEU scores
    """
    try:
        from nltk.translate.bleu_score import corpus_bleu
        
        # Tokenize
        ref_tokens = [[ref.split() for ref in refs] for refs in references]
        hyp_tokens = [hyp.split() for hyp in hypotheses]
        
        bleu1 = corpus_bleu(ref_tokens, hyp_tokens, weights=(1, 0, 0, 0))
        bleu2 = corpus_bleu(ref_tokens, hyp_tokens, weights=(0.5, 0.5, 0, 0))
        bleu3 = corpus_bleu(ref_tokens, hyp_tokens, weights=(0.33, 0.33, 0.33, 0))
        bleu4 = corpus_bleu(ref_tokens, hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25))
        
        return {
            'BLEU-1': bleu1,
            'BLEU-2': bleu2,
            'BLEU-3': bleu3,
            'BLEU-4': bleu4
        }
    except Exception as e:
        print(f"⚠️  Error calculating BLEU: {e}")
        return {'BLEU-1': 0, 'BLEU-2': 0, 'BLEU-3': 0, 'BLEU-4': 0}


def calculate_meteor_score(references, hypotheses):
    """
    Calculate METEOR score
    
    Args:
        references: List of reference captions
        hypotheses: List of generated captions
        
    Returns:
        METEOR score
    """
    try:
        from nltk.translate.meteor_score import meteor_score
        
        scores = []
        for ref, hyp in zip(references, hypotheses):
            score = meteor_score([ref[0]], hyp)
            scores.append(score)
        
        return np.mean(scores)
    except Exception as e:
        print(f"⚠️  Error calculating METEOR: {e}")
        return 0.0


def print_sample_captions(model, dataloader, vocab, device, num_samples=5):
    """
    Print sample generated captions for visual inspection
    
    Args:
        model: Image captioning model
        dataloader: DataLoader
        vocab: Vocabulary
        device: Device
        num_samples: Number of samples to print
    """
    model.eval()
    
    with torch. no_grad():
        images, captions = next(iter(dataloader))
        images = images[:num_samples]. to(device)
        captions = captions[:num_samples]
        
        # Generate captions
        generated = model.generate_caption(images, max_length=30)
        
        print("\n" + "="*80)
        print("📝 SAMPLE GENERATED CAPTIONS")
        print("="*80)
        
        for i in range(num_samples):
            # Reference
            ref = vocab.decode_caption(captions[i], skip_special_tokens=True)
            
            # Generated
            gen = vocab.decode_caption(generated[i], skip_special_tokens=True)
            
            print(f"\nSample {i+1}:")
            print(f"  Reference: {ref}")
            print(f"  Generated: {gen}")
            print(f"  {'-'*76}")
        
        print("="*80 + "\n")


def evaluate_and_print_metrics(model, dataloader, vocab, device):
    """
    Full evaluation with all metrics
    
    Args:
        model: Image captioning model
        dataloader: DataLoader
        vocab: Vocabulary
        device: Device
        
    Returns:
        Dictionary with all metrics
    """
    print("\n" + "="*80)
    print("📊 EVALUATING MODEL")
    print("="*80)
    
    # Generate captions
    references, hypotheses = evaluate_model(model, dataloader, vocab, device)
    
    # Calculate metrics
    print("\nCalculating BLEU scores...")
    bleu_scores = calculate_bleu_scores(references, hypotheses)
    
    print("Calculating METEOR score...")
    meteor = calculate_meteor_score(references, hypotheses)
    
    # Print results
    print("\n" + "="*80)
    print("📈 EVALUATION RESULTS")
    print("="*80)
    print(f"BLEU-1:  {bleu_scores['BLEU-1']:.4f}")
    print(f"BLEU-2:  {bleu_scores['BLEU-2']:.4f}")
    print(f"BLEU-3:  {bleu_scores['BLEU-3']:. 4f}")
    print(f"BLEU-4:  {bleu_scores['BLEU-4']:.4f}")
    print(f"METEOR:  {meteor:.4f}")
    print("="*80 + "\n")
    
    # Print sample captions
    print_sample_captions(model, dataloader, vocab, device, num_samples=5)
    
    return {
        **bleu_scores,
        'METEOR': meteor
    }