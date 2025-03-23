import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Union
import numpy as np
import os

class Dataset:
    def __init__(self, raw_data_path, processed_data_path):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.data = None

    def load_raw_data(self):
        import json
        with open(self.raw_data_path, 'r') as file:
            self.data = json.load(file)

    def preprocess_data(self):
        # Implement preprocessing logic here
        # This could include cleaning, formatting, etc.
        pass

    def save_processed_data(self):
        import json
        with open(self.processed_data_path, 'w') as file:
            json.dump(self.data, file)

    def get_data(self):
        return self.data

class LagrangianDataset(Dataset):
    def __init__(
        self, 
        data_file: str,
        tokenizer,
        max_length: int = 128,
        context_max_length: int = 64
    ):
        """
        Dataset for Lagrangian equations
        
        Args:
            data_file: Path to JSON file with processed data
            tokenizer: Tokenizer to use for encoding
            max_length: Maximum sequence length
            context_max_length: Maximum context length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.context_max_length = context_max_length
        
        # Load data
        with open(data_file, 'r') as f:
            self.examples = json.load(f)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Encode input text
        input_encoding = self.tokenizer(
            example['input_text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Encode output text (target)
        output_encoding = self.tokenizer(
            example['output_text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': output_encoding['input_ids'].squeeze(),
            'is_masked': torch.tensor(example['is_masked'], dtype=torch.bool)
        }

def create_dataloaders(
    data_file: str,
    tokenizer,
    batch_size: int = 8,
    max_length: int = 128,
    context_max_length: int = 64,
    val_split: float = 0.1,
    seed: int = 42
):
    """Create train and validation dataloaders"""
    # Load all examples
    with open(data_file, 'r') as f:
        examples = json.load(f)
    
    # Split into train and val
    np.random.seed(seed)
    indices = np.random.permutation(len(examples))
    val_size = int(len(examples) * val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    train_examples = [examples[i] for i in train_indices]
    val_examples = [examples[i] for i in val_indices]
    
    # Write split data to temp files
    os.makedirs("../../data/processed/temp", exist_ok=True)
    with open("../../data/processed/temp/train_examples.json", 'w') as f:
        json.dump(train_examples, f)
    
    with open("../../data/processed/temp/val_examples.json", 'w') as f:
        json.dump(val_examples, f)
    
    # Create datasets
    train_dataset = LagrangianDataset(
        "../../data/processed/temp/train_examples.json",
        tokenizer,
        max_length,
        context_max_length
    )
    
    val_dataset = LagrangianDataset(
        "../../data/processed/temp/val_examples.json",
        tokenizer,
        max_length,
        context_max_length
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size
    )
    
    return train_dataloader, val_dataloader