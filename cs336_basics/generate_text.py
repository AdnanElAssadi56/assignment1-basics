#!/usr/bin/env python3
"""
Text generation script for trained Transformer language models.
"""

import torch
import argparse
import os
import sys
from tokenizer import Tokenizer
from transformer import TransformerLM
from decoder import generate

def load_model_and_tokenizer(checkpoint_path, vocab_path, merges_path, special_tokens, device):
    """Load trained model and tokenizer from checkpoint."""
    print(f"Loading tokenizer from {vocab_path} and {merges_path}")
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=special_tokens)
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model hyperparameters from checkpoint if available
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
    else:
        # Use default config - you may need to adjust these based on your training
        config = {
            'd_model': 512,
            'num_heads': 16,
            'num_layers': 4,
            'd_ff': 1344,
            'vocab_size': len(tokenizer.vocab),
            'context_length': 256,
            'max_seq_len': 256,
            'theta': 10000.0
        }
        print("Warning: No model config found in checkpoint, using defaults")
    
    # Create model
    model = TransformerLM(
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        vocab_size=config['vocab_size'],
        context_length=config['context_length'],
        num_layers=config['num_layers'],
        max_seq_len=config.get('max_seq_len', config['context_length']),
        theta=config.get('theta', 10000.0),
        device=device
    )
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, tokenizer

def generate_text_samples(model, tokenizer, prompts, max_tokens=256, temperature=1.0, top_p=0.9, device='cuda'):
    """Generate text samples from prompts."""
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"\n{'='*60}")
        print(f"Sample {i+1}: Prompt = '{prompt}'")
        print(f"{'='*60}")
        
        try:
            # Generate text
            generated_text = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                device=device
            )
            
            print(f"Generated text:\n{generated_text}")
            
            # Check for fluency and quality
            token_count = len(tokenizer.encode(generated_text))
            
            results.append({
                'prompt': prompt,
                'generated_text': generated_text,
                'token_count': token_count,
                'temperature': temperature,
                'top_p': top_p
            })
            
        except Exception as e:
            print(f"Error generating text for prompt '{prompt}': {e}")
            results.append({
                'prompt': prompt,
                'error': str(e)
            })
    
    return results

def analyze_generation_quality(results):
    """Analyze the quality of generated text."""
    print(f"\n{'='*80}")
    print("GENERATION QUALITY ANALYSIS")
    print(f"{'='*80}")
    
    for i, result in enumerate(results):
        if 'error' in result:
            print(f"Sample {i+1}: FAILED - {result['error']}")
            continue
            
        text = result['generated_text']
        prompt = result['prompt']
        
        print(f"\nSample {i+1} Analysis:")
        print(f"Prompt: '{prompt}'")
        print(f"Token count: {result['token_count']}")
        
        # Basic fluency checks
        fluency_score = 0
        issues = []
        
        # Check for repetition
        words = text.lower().split()
        if len(words) > 10:
            # Check for excessive repetition
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            max_freq = max(word_freq.values())
            if max_freq > len(words) * 0.3:  # More than 30% repetition
                issues.append("Excessive word repetition")
            else:
                fluency_score += 1
        
        # Check for coherence (basic sentence structure)
        sentences = text.split('.')
        if len(sentences) > 1:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            if 5 <= avg_sentence_length <= 25:  # Reasonable sentence length
                fluency_score += 1
            else:
                issues.append("Unusual sentence lengths")
        
        # Check for proper capitalization
        if text and text[0].isupper():
            fluency_score += 1
        else:
            issues.append("No initial capitalization")
        
        # Check for story-like structure (for TinyStories)
        story_indicators = ['once upon a time', 'the end', 'one day', 'then', 'finally']
        if any(indicator in text.lower() for indicator in story_indicators):
            fluency_score += 1
        
        print(f"Fluency score: {fluency_score}/4")
        if issues:
            print(f"Issues: {', '.join(issues)}")
        else:
            print("No major issues detected")

def main():
    parser = argparse.ArgumentParser(description="Generate text from trained Transformer LM")
    
    # Model and tokenizer paths
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to model checkpoint (.pt file)")
    parser.add_argument("--vocab_path", type=str, 
                       default="outputs/tinystories_bpe_10k/vocab.pkl",
                       help="Path to vocabulary file")
    parser.add_argument("--merges_path", type=str,
                       default="outputs/tinystories_bpe_10k/merges.pkl", 
                       help="Path to merges file")
    parser.add_argument("--special_tokens", type=str, nargs='+', 
                       default=["<|endoftext|>"],
                       help="Special tokens")
    
    # Generation parameters
    parser.add_argument("--prompts", type=str, nargs='+',
                       default=["Once upon a time", "There was a little girl", "In a magical forest"],
                       help="Prompts for text generation")
    parser.add_argument("--max_tokens", type=int, default=256,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--device", type=str, 
                       default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use")
    
    # Output options
    parser.add_argument("--output_file", type=str, default=None,
                       help="Save generated text to file")
    parser.add_argument("--num_samples", type=int, default=3,
                       help="Number of samples to generate per prompt")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint file not found: {args.checkpoint_path}")
        sys.exit(1)
    
    if not os.path.exists(args.vocab_path):
        print(f"Error: Vocabulary file not found: {args.vocab_path}")
        sys.exit(1)
        
    if not os.path.exists(args.merges_path):
        print(f"Error: Merges file not found: {args.merges_path}")
        sys.exit(1)
    
    # Load model and tokenizer
    try:
        model, tokenizer = load_model_and_tokenizer(
            args.checkpoint_path, args.vocab_path, args.merges_path, 
            args.special_tokens, args.device
        )
        print(f"Successfully loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Generate text samples
    all_results = []
    for sample_idx in range(args.num_samples):
        print(f"\n{'#'*80}")
        print(f"GENERATING SAMPLE SET {sample_idx + 1}/{args.num_samples}")
        print(f"{'#'*80}")
        
        results = generate_text_samples(
            model, tokenizer, args.prompts, 
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=args.device
        )
        all_results.extend(results)
    
    # Analyze generation quality
    analyze_generation_quality(all_results)
    
    # Save to file if requested
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(f"Text Generation Results\n")
            f.write(f"Checkpoint: {args.checkpoint_path}\n")
            f.write(f"Temperature: {args.temperature}, Top-p: {args.top_p}\n")
            f.write(f"Max tokens: {args.max_tokens}\n\n")
            
            for i, result in enumerate(all_results):
                if 'error' in result:
                    f.write(f"Sample {i+1}: ERROR - {result['error']}\n\n")
                else:
                    f.write(f"Sample {i+1}:\n")
                    f.write(f"Prompt: {result['prompt']}\n")
                    f.write(f"Generated ({result['token_count']} tokens):\n")
                    f.write(f"{result['generated_text']}\n\n")
        
        print(f"\nResults saved to: {args.output_file}")

if __name__ == "__main__":
    main()