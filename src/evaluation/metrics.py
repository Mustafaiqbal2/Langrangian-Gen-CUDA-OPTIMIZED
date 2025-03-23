import torch
import numpy as np
from sklearn.metrics import accuracy_score
import sympy as sp
from sympy.parsing.latex import parse_latex
from typing import List, Dict, Tuple
import re
import os
import sys
from pathlib import Path

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.transformer import LagrangianTransformer
from transformers import AutoTokenizer

def clean_equation(eq: str) -> str:
    """Clean generated equation text for comparison"""
    # Remove leading/trailing whitespace
    eq = eq.strip()
    
    # Extract just the Lagrangian part if it contains context
    if ':' in eq:
        eq = eq.split(':')[-1].strip()
    
    # If the equation starts with "L = ", keep only what follows
    if eq.startswith("L = "):
        eq = eq[4:].strip()
    
    # Replace multiple spaces with single space
    eq = re.sub(r'\s+', ' ', eq)
    
    return eq

def equation_similarity(eq1: str, eq2: str) -> float:
    """
    Compute similarity between two equations
    Simple implementation based on token overlap
    """
    eq1 = clean_equation(eq1)
    eq2 = clean_equation(eq2)
    
    # Tokenize equations (simple whitespace tokenization)
    tokens1 = set(eq1.split())
    tokens2 = set(eq2.split())
    
    # Calculate Jaccard similarity
    if not tokens1 and not tokens2:
        return 1.0  # Both empty
    
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    
    return intersection / union

def exact_match_accuracy(predictions: List[str], targets: List[str]) -> float:
    """Compute exact match accuracy"""
    cleaned_preds = [clean_equation(pred) for pred in predictions]
    cleaned_targets = [clean_equation(target) for target in targets]
    
    matches = sum(1 for pred, target in zip(cleaned_preds, cleaned_targets) if pred == target)
    return matches / len(predictions) if predictions else 0.0

def term_based_accuracy(predictions: List[str], targets: List[str]) -> float:
    """Compute term-based accuracy"""
    similarities = [equation_similarity(pred, target) for pred, target in zip(predictions, targets)]
    return np.mean(similarities)

def evaluate_model(
    model_path: str,
    test_data_path: str,
    output_dir: str = None
) -> Dict:
    """
    Evaluate a trained model on test data
    
    Args:
        model_path: Path to trained model
        test_data_path: Path to test data JSON file
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Load model and tokenizer
    transformer = LagrangianTransformer()
    transformer.load_model(model_path)
    model = transformer.model
    tokenizer = transformer.tokenizer
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Load test data
    import json
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    # Generate predictions
    predictions = []
    targets = []
    
    for example in test_data:
        input_text = example['input_text']
        target_text = example['output_text']
        
        # Generate prediction
        pred = transformer.generate_lagrangian(
            input_text,
            max_length=128,
            temperature=0.7,
            num_return_sequences=1
        )[0]
        
        predictions.append(pred)
        targets.append(target_text)
    
    # Calculate metrics
    exact_match = exact_match_accuracy(predictions, targets)
    term_accuracy = term_based_accuracy(predictions, targets)
    
    # Save results
    results = {
        "exact_match_accuracy": exact_match,
        "term_based_accuracy": term_accuracy,
        "predictions": predictions,
        "targets": targets
    }
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "evaluation_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
    
    return results

def main():
    # Set paths
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "best_model")
    test_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed", "training_examples.json")
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")
    
    # Evaluate model
    results = evaluate_model(model_path, test_data_path, output_dir)
    
    # Print results
    print("Evaluation Results:")
    print(f"Exact Match Accuracy: {results['exact_match_accuracy']:.4f}")
    print(f"Term-based Accuracy: {results['term_based_accuracy']:.4f}")

if __name__ == "__main__":
    main()