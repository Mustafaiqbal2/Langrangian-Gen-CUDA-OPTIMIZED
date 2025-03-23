import re
from typing import List, Dict, Tuple, Union
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
import numpy as np
import sympy as sp
from sympy.parsing.latex import parse_latex

class MathTokenizer:
    """Special tokenizer for mathematical equations and Lagrangians"""
    
    MATH_OPERATORS = ['+', '-', '*', '/', '=', '^', '_', '\\partial', '\\nabla', 
                      '\\sqrt', '\\frac', '\\int', '\\sum', '\\prod']
    
    GREEK_LETTERS = ['\\alpha', '\\beta', '\\gamma', '\\Gamma', '\\delta', '\\Delta', 
                     '\\epsilon', '\\varepsilon', '\\zeta', '\\eta', '\\theta', 
                     '\\Theta', '\\iota', '\\kappa', '\\lambda', '\\Lambda', 
                     '\\mu', '\\nu', '\\xi', '\\Xi', '\\pi', '\\Pi', '\\rho', 
                     '\\sigma', '\\Sigma', '\\tau', '\\upsilon', '\\Upsilon', 
                     '\\phi', '\\Phi', '\\chi', '\\psi', '\\Psi', '\\omega', '\\Omega']
    
    PHYSICS_SYMBOLS = ['m', 'g', 'h', 'k', 'x', 'v', 'F', 'L', 'T', 'V', 'q', 
                       'A', 'φ', 'R', 'ψ', '\\psi', '\\phi']
    
    def __init__(self, base_tokenizer_name: str = "gpt2"):
        """Initialize with a base tokenizer and add special tokens"""
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name)
        
        # Add special tokens for math and physics
        special_tokens = []
        special_tokens.extend(self.MATH_OPERATORS)
        special_tokens.extend(self.GREEK_LETTERS)
        special_tokens.extend(self.PHYSICS_SYMBOLS)
        
        # Add common physics operators and patterns
        special_tokens.extend([
            '\\dot{', '}', 
            '\\ddot{', 
            '\\vec{',
            '\\mathcal{',
            '\\mathrm{',
            '\\text{',
            '\\left(', '\\right)',
            '\\left[', '\\right]',
            '\\begin{equation}', '\\end{equation}'
        ])
        
        # Define special tokens dict
        special_tokens_dict = {'additional_special_tokens': special_tokens}
        
        # Add special tokens to the tokenizer
        num_added_toks = self.base_tokenizer.add_special_tokens(special_tokens_dict)
        print(f"Added {num_added_toks} special tokens for mathematical notation")
        
        # Ensure padding token is set
        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
        
        # Flag indicating if tensor outputs should be returned
        self.return_tensors = None
    
    def pre_tokenize_equation(self, equation: str) -> str:
        """Pre-process an equation for better tokenization"""
        # Add spaces around operators to help tokenization
        for op in ['+', '-', '=', '*', '/', '^', '(', ')', '[', ']', '{', '}']:
            equation = equation.replace(op, f" {op} ")
        
        # Fix consecutive spaces
        equation = re.sub(r'\s+', ' ', equation).strip()
        
        return equation
    
    def tokenize_equation(self, equation: str) -> List[str]:
        """Tokenize an equation string to token list, with special handling for LaTeX"""
        # Pre-process equation
        processed_eq = self.pre_tokenize_equation(equation)
        
        # Perform base tokenization
        tokens = self.base_tokenizer.tokenize(processed_eq)
        
        return tokens
    
    def encode_equation(
        self, 
        equation: str, 
        max_length: int = None,
        padding: str = None,
        truncation: bool = None,
        return_tensors: str = None
    ) -> Union[Dict[str, torch.Tensor], Dict[str, List]]:
        """Encode an equation string to token IDs"""
        # Store return_tensors flag for decode method
        self.return_tensors = return_tensors
        
        # Pre-process equation
        processed_eq = self.pre_tokenize_equation(equation)
        
        # Encode using base tokenizer
        encoding = self.base_tokenizer(
            processed_eq,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors
        )
        
        return encoding
    
    def decode_equation(
        self, 
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True
    ) -> str:
        """Decode token IDs back to an equation string"""
        # If tensor was returned during encoding, ensure we handle it properly
        if isinstance(token_ids, torch.Tensor):
            if token_ids.dim() > 1 and token_ids.shape[0] == 1:
                token_ids = token_ids.squeeze(0)
            token_ids = token_ids.tolist()
        
        # Decode using base tokenizer
        decoded = self.base_tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )
        
        # Post-process: remove extra spaces around operators
        for op in ['+', '-', '=', '*', '/', '^']:
            decoded = decoded.replace(f" {op} ", op)
        
        # Clean up brackets and parentheses
        for pair in [('( ', '('), (' )', ')'), ('[ ', '['), (' ]', ']')]:
            decoded = decoded.replace(pair[0], pair[1])
        
        return decoded
    
    def __call__(
        self,
        text: Union[str, List[str]],
        max_length: int = None,
        padding: str = None,
        truncation: bool = None,
        return_tensors: str = None
    ) -> Dict:
        """Make the tokenizer callable just like HuggingFace tokenizers"""
        # Handle lists of texts
        if isinstance(text, list):
            return {
                key: [val[i] for i in range(len(text))] if not isinstance(val, torch.Tensor) else val
                for key, val in self.base_tokenizer(
                    text, 
                    max_length=max_length,
                    padding=padding,
                    truncation=truncation,
                    return_tensors=return_tensors
                ).items()
            }
        
        # Check if this is likely an equation or LaTeX
        if any(symbol in text for symbol in ['\\', '^', '_', '=', 'L = ']):
            return self.encode_equation(
                text,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                return_tensors=return_tensors
            )
        
        # If not an equation, use regular tokenizer
        return self.base_tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors
        )
    
    def save_pretrained(self, save_directory: str):
        """Save the tokenizer to a directory"""
        self.base_tokenizer.save_pretrained(save_directory)
    
    def analyze_equation(self, equation_str):
        """Analyze math equation with proper empty handling"""
        # Always initialize with required structure
        result = {
            "valid_parse": False,
            "tokens": [],
            "potential_symbols": []
        }
        
        # Handle empty input case explicitly
        if not equation_str or equation_str.strip() == "":
            result["error"] = "Empty equation"
            return result
        
        # Continue with regular parsing for non-empty cases
        # Initialize result structure
        result = {
            "valid_parse": False,
            "tokens": [],
            "potential_symbols": []
        }
        
        # Handle empty input explicitly
        if not equation_str or equation_str.strip() == "":
            result["error"] = "Empty equation"
            return result
        
        # Clean up the equation
        cleaned = equation_str.strip()
        if cleaned.startswith("L = "):
            cleaned = cleaned[4:].strip()
        
        # If it's still empty after cleaning
        if not cleaned:
            result["error"] = "Empty equation after cleaning"
            return result
        
        # Tokenize first (this always works)
        tokens = cleaned.replace('^', ' ^ ').replace('·', ' · ').replace('*', ' * ')
        tokens = tokens.replace('+', ' + ').replace('-', ' - ').replace('/', ' / ')
        tokens = tokens.replace('(', ' ( ').replace(')', ' ) ')
        tokens = [t for t in tokens.split() if t.strip()]
        result["tokens"] = tokens
        
        # Extract potential symbols
        result["potential_symbols"] = [t for t in tokens if t.isalpha() and len(t)==1]
        
        try:
            # Convert to sympy-friendly syntax
            # First handle fractions
            processed = re.sub(r'\(\s*(\d+)\s*/\s*(\d+)\s*\)', 
                              lambda m: str(float(int(m.group(1)))/float(int(m.group(2)))), 
                              cleaned)
            
            # Handle operators
            processed = processed.replace('^', '**').replace('·', '*')
            
            # Add implied multiplication
            processed = re.sub(r'(\d+|\))([a-zA-Z])', r'\1*\2', processed)
            processed = re.sub(r'([a-zA-Z])([a-zA-Z])', r'\1*\2', processed)
            processed = re.sub(r'([a-zA-Z])(\()', r'\1*\2', processed)
            
            # Parse with sympy
            expr = sp.sympify(processed)
            
            # Extract information
            symbols = list(expr.free_symbols)
            
            # Get terms
            if isinstance(expr, sp.Add):
                terms = expr.args
            else:
                terms = [expr]
            
            # Update result with success data
            result["valid_parse"] = True
            result["symbolic_expr"] = expr
            result["symbols"] = symbols
            result["term_count"] = len(terms)
            result["terms"] = terms
            
            # Also add expanded form
            result["expanded_expr"] = sp.expand(expr)
            
        except Exception as e:
            # Provide detailed error information
            result["error"] = f"Parse error: {str(e)}"
            result["processed"] = processed if 'processed' in locals() else "Processing failed"
        
        return result