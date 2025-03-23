import os
import json
import torch
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import sys
import yaml

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.transformer import LagrangianTransformer
from src.data.dataset import create_dataloaders
from src.data.preprocessing import LagrangianPreprocessor
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train_model(
    model_name: str = "gpt2",
    data_dir: str = "../../data",
    output_dir: str = "../../models",
    batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 5e-5,
    max_length: int = 128,
    context_max_length: int = 64,
    warmup_steps: int = 100,
    from_pretrained: bool = True,
    save_every: int = 1,
    use_cuda: bool = True
):
    """Train the Lagrangian Generator model"""
    # Initialize tokenizer and preprocessing
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Check if CUDA is available
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU for training")
    
    # Prepare data if not already done
    preprocessor = LagrangianPreprocessor()
    raw_data_path = os.path.join(data_dir, "raw", "physics_equations.json")
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    processed_data_path = os.path.join(processed_dir, "tokenized_equations.json")
    training_examples_path = os.path.join(processed_dir, "training_examples.json")
    
    if not os.path.exists(processed_data_path):
        df = preprocessor.prepare_dataset(raw_data_path, processed_data_path)
    
    if not os.path.exists(training_examples_path):
        df = preprocessor.prepare_dataset(raw_data_path)
        training_examples = preprocessor.create_training_examples(df)
        
        with open(training_examples_path, 'w') as f:
            json.dump(training_examples, f, indent=2)
    
    # Create data loaders
    train_dataloader, val_dataloader = create_dataloaders(
        training_examples_path,
        tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        context_max_length=context_max_length
    )
    
    # Initialize model
    transformer = LagrangianTransformer(model_name, from_pretrained=from_pretrained)
    model = transformer.model
    model.to(device)
    
    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_dataloader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )
            
            loss = outputs.loss
            train_loss += loss.item()
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_train_loss = train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0
        progress_bar = tqdm(val_dataloader, desc="Validation")
        
        with torch.no_grad():
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    labels=labels
                )
                
                loss = outputs.loss
                val_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({"loss": loss.item()})
        
        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        print(f"Average validation loss: {avg_val_loss:.4f}")
        
        # Save model if it's the best so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path = os.path.join(output_dir, "best_model")
            transformer.save_model(model_save_path)
            print(f"Best model saved to {model_save_path}")
        
        # Save model checkpoint
        if (epoch + 1) % save_every == 0:
            model_save_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}")
            transformer.save_model(model_save_path)
            print(f"Checkpoint saved to {model_save_path}")
    
    # Save final model
    model_save_path = os.path.join(output_dir, "final_model")
    transformer.save_model(model_save_path)
    print(f"Final model saved to {model_save_path}")
    
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()
    
    # Return training history
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss
    }

def main():
    # Create output directories
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Train model with CUDA optimizations if available
    history = train_model(
        model_name="gpt2",
        data_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"),
        output_dir=model_dir,
        batch_size=4,  # Adjust based on your GPU memory
        num_epochs=5,
        learning_rate=5e-5,
        use_cuda=True
    )
    
    print("Training completed!")
    print(f"Best validation loss: {history['best_val_loss']:.4f}")

if __name__ == "__main__":
    main()