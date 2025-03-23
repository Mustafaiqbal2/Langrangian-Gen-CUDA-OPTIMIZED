import json
import os
import sympy as sp
from sympy.parsing.latex import parse_latex
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union

class LagrangianPreprocessor:
    def __init__(self, config_path=None):
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                import yaml
                self.config = yaml.safe_load(f)
    
    def load_equations(self, file_path: str) -> List[Dict]:
        """Load equations from JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data['equations']
    
    def parse_latex_equation(self, equation: str) -> sp.Expr:
        """Parse LaTeX equation to SymPy expression"""
        try:
            # Clean LaTeX string for parsing
            cleaned_eq = equation.replace('L = ', '')
            expr = parse_latex(cleaned_eq)
            return expr
        except Exception as e:
            print(f"Error parsing equation: {equation}")
            print(f"Error details: {e}")
            return None
    
    def equation_to_features(self, equation_data: Dict) -> Dict:
        """Convert equation data to features for model input"""
        name = equation_data['name']
        context = equation_data['context']
        equation = equation_data['equation']
        
        # Extract the right side of the equation (after L =)
        if 'L = ' in equation:
            equation_terms = equation.split('L = ')[1]
        else:
            equation_terms = equation
        
        # Tokenize the equation (simple space-based tokenization for now)
        tokens = equation_terms.split()
        
        return {
            'name': name,
            'context': context,
            'equation': equation,
            'equation_terms': equation_terms,
            'tokens': tokens
        }
    
    def prepare_dataset(self, raw_data_path: str, output_path: str = None) -> pd.DataFrame:
        """Process raw data and convert to model-ready dataset"""
        equations = self.load_equations(raw_data_path)
        processed_data = []
        
        for eq in equations:
            features = self.equation_to_features(eq)
            processed_data.append(features)
        
        df = pd.DataFrame(processed_data)
        
        if output_path:
            df.to_json(output_path, orient='records')
            print(f"Processed data saved to {output_path}")
        
        return df
    
    def create_training_examples(self, df: pd.DataFrame, mask_prob: float = 0.3) -> List[Dict]:
        """Create training examples by masking parts of equations"""
        training_examples = []
        
        for _, row in df.iterrows():
            tokens = row['tokens']
            n_tokens = len(tokens)