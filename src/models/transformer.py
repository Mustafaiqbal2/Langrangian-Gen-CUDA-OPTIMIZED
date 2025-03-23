import torch
import torch.nn as nn
from transformers import (
    GPT2LMHeadModel, 
    GPT2Config,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel
)
from typing import Dict, List, Optional, Union, Tuple
import os
import yaml

class LagrangianTransformer:
    def __init__(
        self,
        model_name: str = "gpt2",
        config_path: Optional[str] = None,
        from_pretrained: bool = True
    ):
        """
        Lagrangian Generator based on transformer architecture
        
        Args:
            model_name: Name of the model or path to model directory
            config_path: Path to config file (YAML)
            from_pretrained: Whether to load pretrained weights
        """
        self.model_name = model_name
        self.config = {}
        
        # Load config if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model
        if from_pretrained:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            # Create custom config if not loading pretrained
            model_config = GPT2Config(
                vocab_size=self.tokenizer.vocab_size,
                n_positions=128,
                n_ctx=128,
                n_embd=256,
                n_layer=6,
                n_head=8
            )
            self.model = GPT2LMHeadModel(model_config)
    
    def save_model(self, output_dir: str):
        """Save model and tokenizer to directory"""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")
    
    def load_model(self, model_dir: str):
        """Load model and tokenizer from directory"""
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        print(f"Model loaded from {model_dir}")
    
    def generate_lagrangian(
        self, 
        context: str,
        max_length: int = 100,
        temperature: float = 0.7,
        num_return_sequences: int = 1
    ) -> List[str]:
        """Generate Lagrangian given a context"""
        # Add a better prompt format to guide the model
        formatted_prompt = f"Given the following physics context, generate the corresponding Lagrangian equation:\nContext: {context}\nLagrangian equation: L = "
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            self.model = self.model.cuda()
        
        # Generate text
        outputs = self.model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],  # Explicitly pass attention mask
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
        
        # Decode and return
        generated_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return generated_texts

class CustomTransformerModel(nn.Module):
    """A smaller custom transformer for Lagrangian generation"""
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(512, d_model)  # Position encoding
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Output layer
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)
    
    def forward(
        self, 
        src: torch.Tensor, 
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            src: Input tensor [batch_size, seq_len]
            src_mask: Mask for padding [batch_size, seq_len]
            
        Returns:
            Output tensor [batch_size, seq_len, vocab_size]
        """
        # Create position indices
        positions = torch.arange(0, src.size(1), device=src.device).unsqueeze(0).expand(src.size(0), -1)
        
        # Embed tokens and positions
        src = self.embedding(src) + self.pos_encoder(positions)
        
        # Apply transformer
        if src_mask is not None:
            src = src * src_mask.unsqueeze(-1)
        
        output = self.transformer_encoder(src)
        output = self.fc_out(output)
        
        return output