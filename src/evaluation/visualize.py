import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import os
from transformers import AutoTokenizer
import matplotlib.ticker as ticker

def plot_lagrangian_terms(terms, title='Lagrangian Terms Visualization'):
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(terms)), terms.values(), align='center')
    plt.xticks(range(len(terms)), list(terms.keys()), rotation=45)
    plt.title(title)
    plt.xlabel('Terms')
    plt.ylabel('Values')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

def plot_accuracy(history, title='Model Accuracy'):
    plt.figure(figsize=(10, 6))
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_loss(history, title='Model Loss'):
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

class AttentionVisualizer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def get_attention_weights(self, text: str) -> Tuple[torch.Tensor, List[str]]:
        """Extract attention weights from the model for a given input"""
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Move to the same device as model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Get token list for visualization
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        # Forward pass with output_attentions=True to get attention weights
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=True,
                return_dict=True
            )
        
        # Get attention weights
        # Shape: [num_layers, batch_size, num_heads, seq_len, seq_len]
        attention = outputs.attentions
        
        return attention, tokens
    
    def plot_attention_heads(
        self, 
        attention: torch.Tensor, 
        tokens: List[str], 
        layer: int = -1, 
        heads: Optional[List[int]] = None,
        output_dir: str = "results/attention_plots",
        filename: str = "attention_heads.png"
    ):
        """Plot attention weights for multiple heads in a grid"""
        # If no specific heads are specified, plot all heads
        if heads is None:
            heads = list(range(attention[layer].shape[1]))
        
        # Create a grid of subplots
        fig, axs = plt.subplots(
            1, len(heads), 
            figsize=(4 * len(heads), 4), 
            sharey=True
        )
        
        # Handle the case where there's only one head
        if len(heads) == 1:
            axs = [axs]
        
        # Plot each attention head
        for i, head_idx in enumerate(heads):
            head_attention = attention[layer][0, head_idx].cpu().numpy()
            ax = axs[i]
            im = ax.imshow(head_attention, cmap="viridis")
            
            # Set labels
            ax.set_title(f"Head {head_idx}")
            
            # Show token labels sparingly if there are many tokens
            if len(tokens) > 20:
                # Only show some tokens to avoid overcrowding
                indices = np.linspace(0, len(tokens)-1, 10).astype(int)
                ax.set_xticks(indices)
                ax.set_xticklabels([tokens[i] for i in indices], rotation=90)
                ax.set_yticks(indices)
                ax.set_yticklabels([tokens[i] for i in indices])
            else:
                ax.set_xticks(range(len(tokens)))
                ax.set_xticklabels(tokens, rotation=90)
                ax.set_yticks(range(len(tokens)))
                ax.set_yticklabels(tokens)
        
        plt.tight_layout()
        
        # Save plot if output directory is specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_attention_map(
        self,
        text: str,
        layer: int = -1,
        head: int = 0,
        output_dir: str = "results/attention_plots",
        filename: str = "attention_map.png",
        show_plot: bool = True
    ):
        """Generate and plot attention map with improved error handling"""
        try:
            attention, tokens = self.get_attention_weights(text)
            
            # Validate indices
            if abs(layer) >= len(attention):
                layer = -1
                print(f"Warning: Layer index out of range. Using last layer.")
                
            num_heads = attention[layer].shape[1]
            if head >= num_heads:
                head = 0
                print(f"Warning: Head index out of range. Using first head.")
            
            # Get attention matrix
            attn_matrix = attention[layer][0, head].cpu().numpy()
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 10))
            im = ax.imshow(attn_matrix, cmap="YlOrRd")
            
            # Add colorbar
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel("Attention Weight", rotation=-90, va="bottom")
            
            # Handle token display for readability
            if len(tokens) > 30:
                # For many tokens, show a subset
                step = max(1, len(tokens) // 30)
                indices = list(range(0, len(tokens), step))
                if indices[-1] != len(tokens) - 1:
                    indices.append(len(tokens) - 1)
                    
                ax.set_xticks(indices)
                ax.set_xticklabels([tokens[i] for i in indices], rotation=90, fontsize=8)
                ax.set_yticks(indices)
                ax.set_yticklabels([tokens[i] for i in indices], fontsize=8)
            else:
                ax.set_xticks(range(len(tokens)))
                ax.set_xticklabels(tokens, rotation=90)
                ax.set_yticks(range(len(tokens)))
                ax.set_yticklabels(tokens)
            
            plt.title(f"Attention Map (Layer {layer}, Head {head})")
            plt.tight_layout()
            
            # Save and show
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            
            if show_plot:
                plt.show()
            else:
                plt.close(fig)
                
            return fig
        except Exception as e:
            print(f"Error plotting attention map: {e}")
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(0.5, 0.5, f"Attention visualization failed: {str(e)}", 
                    ha='center', va='center')
            ax.axis('off')
            return fig
    
    def visualize_equation_attention(
        self,
        physics_context: str,
        generated_equation: str,
        output_dir: str = "results/attention_plots"
    ):
        """Analyze which parts of the context influence specific terms in the equation"""
        # Format input as used during generation
        input_text = f"Given the following physics context, generate the corresponding Lagrangian equation:\nContext: {physics_context}\nLagrangian equation: L = "
        
        # Get attention weights
        attention, input_tokens = self.get_attention_weights(input_text)
        
        # Parse generated equation (without the context)
        equation_tokens = self.tokenizer.tokenize(generated_equation)
        
        # Find where context and equation parts are in the input
        context_indices = []
        for i, token in enumerate(input_tokens):
            if token in self.tokenizer.tokenize(physics_context):
                context_indices.append(i)
        
        # Create average attention from all layers and heads
        avg_attention = torch.mean(torch.mean(attention[-1], dim=0), dim=0).cpu().numpy()
        
        # Create figure showing how physics context influences equation terms
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Extract the relevant part of the attention matrix
        # (from context tokens to generated tokens)
        if len(context_indices) > 0:
            # Simplified: just show attention from context to last token
            context_to_last = avg_attention[-1, context_indices]
            
            # Plot as bar chart
            ax.bar(range(len(context_indices)), context_to_last)
            ax.set_xticks(range(len(context_indices)))
            ax.set_xticklabels([input_tokens[i] for i in context_indices], rotation=90)
            ax.set_ylabel("Attention Weight")
            ax.set_title(f"How Context Influences Last Generated Token")
        
        plt.tight_layout()
        
        # Save figure
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, "context_influence.png"), dpi=300, bbox_inches='tight')
        
        return fig