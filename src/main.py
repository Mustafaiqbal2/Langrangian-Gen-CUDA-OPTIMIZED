import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).parent.parent))

from src.models.transformer import LagrangianTransformer
from src.data.preprocessing import LagrangianPreprocessor
from src.training.train import train_model
from src.evaluation.metrics import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Neural Lagrangian Generator CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess raw data")
    preprocess_parser.add_argument("--input", type=str, default="data/raw/physics_equations.json", help="Input raw data file")
    preprocess_parser.add_argument("--output", type=str, default="data/processed/tokenized_equations.json", help="Output processed data file")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--model", type=str, default="gpt2", help="Model name or path")
    train_parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    train_parser.add_argument("--output-dir", type=str, default="models", help="Output directory for model")
    train_parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    train_parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    train_parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    train_parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA")
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate Lagrangian")
    generate_parser.add_argument("--model-path", type=str, default="models/best_model", help="Path to trained model")
    generate_parser.add_argument("--prompt", type=str, required=True, help="Input prompt for generation")
    generate_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    generate_parser.add_argument("--max-length", type=int, default=128, help="Maximum length for generation")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    eval_parser.add_argument("--model-path", type=str, default="models/best_model", help="Path to trained model")
    eval_parser.add_argument("--test-data", type=str, default="data/processed/training_examples.json", help="Test data file")
    eval_parser.add_argument("--output-dir", type=str, default="results", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Handle commands
    if args.command == "preprocess":
        print(f"Preprocessing data from {args.input} to {args.output}")
        preprocessor = LagrangianPreprocessor()
        df = preprocessor.prepare_dataset(args.input, args.output)
        training_examples = preprocessor.create_training_examples(df)
        
        # Save training examples
        training_output = os.path.join(os.path.dirname(args.output), "training_examples.json")
        import json
        with open(training_output, 'w') as f:
            json.dump(training_examples, f, indent=2)
        
        print(f"Created {len(training_examples)} training examples")
        print(f"Saved to {training_output}")
    
    elif args.command == "train":
        print(f"Training model {args.model}")
        history = train_model(
            model_name=args.model,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            use_cuda=not args.no_cuda
        )
        
        print("Training completed!")
        print(f"Best validation loss: {history['best_val_loss']:.4f}")
    
    elif args.command == "generate":
        print(f"Generating Lagrangian using model from {args.model_path}")
        transformer = LagrangianTransformer()
        transformer.load_model(args.model_path)
        
        generated = transformer.generate_lagrangian(
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            num_return_sequences=1
        )[0]
        
        print("\nGenerated Lagrangian:")
        print("-" * 40)
        print(generated)
        print("-" * 40)
    
    elif args.command == "evaluate":
        print(f"Evaluating model from {args.model_path}")
        results = evaluate_model(
            model_path=args.model_path,
            test_data_path=args.test_data,
            output_dir=args.output_dir
        )
        
        print("Evaluation Results:")
        print(f"Exact Match Accuracy: {results['exact_match_accuracy']:.4f}")
        print(f"Term-based Accuracy: {results['term_based_accuracy']:.4f}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()