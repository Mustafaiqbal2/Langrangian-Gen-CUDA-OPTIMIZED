import re
import sympy as sp
from typing import Dict, List, Union, Optional

class PhysicsEvaluator:
    """Evaluates Lagrangians against known physics principles"""
    
    @staticmethod
    def check_lagrangian(lagrangian: Union[str, sp.Expr], domain: str = "classical") -> Dict:
        """Fixed to handle domain name variations and parsing physics notation"""
        # Standard result initialization
        result = {
            "valid": False,
            "properties": [],
            "score": 0,
            "suggestions": []
        }
        
        try:
            # Handle empty input
            if isinstance(lagrangian, str) and not lagrangian.strip():
                result["error"] = "Empty Lagrangian"
                return result
                
            # Parse string input if needed
            if isinstance(lagrangian, str):
                try:
                    # Format physics notation to sympy-compatible syntax
                    formatted_eq = lagrangian.replace('^', '**').replace('·', '*')
                    
                    # Handle fractions like (1/2)
                    formatted_eq = re.sub(r'\(\s*(\d+)\s*/\s*(\d+)\s*\)', 
                                  lambda m: str(float(int(m.group(1)))/float(int(m.group(2)))), 
                                  formatted_eq)
                    
                    # Handle implicit multiplication (a b → a*b)
                    formatted_eq = re.sub(r'([a-zA-Z0-9])[\s]*([a-zA-Z])', r'\1*\2', formatted_eq)
                    
                    # Clean up any remaining whitespace
                    formatted_eq = formatted_eq.replace(' ', '')
                    
                    # Now parse with sympy
                    lagrangian = sp.sympify(formatted_eq)
                    result["valid"] = True
                except Exception as e:
                    # If parsing fails, still continue with the original string
                    result["error"] = f"Parse error: {str(e)}"
                    # For physics evaluation, convert to simplified expression
                    lagrangian = sp.sympify("0")  # Default to zero if parsing fails
            
            # Map domain names here - this is the fix
            if domain == "classical_mechanics":
                domain = "classical"
            
            # Handle domain-specific evaluation
            if domain == "electromagnetic":
                domain_result = PhysicsEvaluator._check_electromagnetic(lagrangian)
            elif domain == "classical" or domain == "classical_mechanics":
                domain_result = PhysicsEvaluator._check_classical(lagrangian)
            else:
                domain_result = PhysicsEvaluator._check_generic(lagrangian)
            
            # Update result with domain-specific findings
            result.update({
                "score": domain_result.get("score", 0),
                "properties": domain_result.get("properties", []),
                "suggestions": domain_result.get("suggestions", [])
            })
            
        except Exception as e:
            result["error"] = str(e)
            
        return result

    @staticmethod
    def _check_classical(expr: sp.Expr) -> Dict:
        """Check classical mechanics Lagrangian properties"""
        result = {"properties": [], "score": 0, "suggestions": []}
        
        # Get variables & expressions
        variables = [str(s) for s in expr.free_symbols]
        expanded = sp.expand(expr)
        expr_str = str(expanded)
        
        # Check for kinetic term T = (1/2)mv²
        has_m = 'm' in variables
        has_v = 'v' in variables
        has_kinetic = has_m and has_v and ('m*v**2' in expr_str or '0.5*m*v**2' in expr_str)
        
        if has_kinetic:
            result["properties"].append("has_kinetic_energy")
            result["score"] += 20
        else:
            result["suggestions"].append("Missing kinetic energy term")
        
        # Check for spring potential
        has_k = 'k' in variables
        has_x = 'x' in variables
        has_spring = has_k and has_x and ('k*x**2' in expr_str or '0.5*k*x**2' in expr_str)
        
        # Check for gravitational
        has_g = 'g' in variables
        has_h = 'h' in variables
        has_grav = has_m and has_g and has_h and ('m*g*h' in expr_str or 'g*m*h' in expr_str)
        
        if has_spring:
            result["properties"].append("has_spring_term")
            result["properties"].append("has_potential_energy")
            result["score"] += 20
        
        if has_grav:
            result["properties"].append("has_gravitational_term")
            result["properties"].append("has_potential_energy")
            result["score"] += 20
        
        return result

    @staticmethod
    def _check_generic(expr: sp.Expr) -> Dict:
        """Generic physics evaluator for domains without specialized handlers"""
        result = {"properties": [], "score": 0, "suggestions": []}
        
        # Get information about the expression
        variables = list(expr.free_symbols)
        var_names = [str(v) for v in variables]
        expanded = sp.expand(expr)
        expr_str = str(expanded)
        
        # Check for quadratic terms (common in physics)
        has_squared = False
        for v in var_names:
            if f"{v}**2" in expr_str:
                has_squared = True
                result["properties"].append(f"has_{v}_squared")
                result["score"] += 10
        
        if has_squared:
            result["properties"].append("has_squared_terms")
        
        # Check for interaction terms (products of different variables)
        for i, v1 in enumerate(var_names):
            for v2 in var_names[i+1:]:
                if f"{v1}*{v2}" in expr_str or f"{v2}*{v1}" in expr_str:
                    result["properties"].append(f"has_{v1}_{v2}_interaction")
                    result["score"] += 10
        
        # Check overall structure (prefer at least 2 variables)
        if len(variables) >= 2:
            result["properties"].append("has_multiple_variables")
            result["score"] += 5
        
        return result

# Safe print helper
def safe_print_properties(result_dict):
    """Print properties safely even if key is missing"""
    if result_dict and isinstance(result_dict, dict):
        properties = result_dict.get("properties", [])
        if properties:
            return ", ".join(properties)
        else:
            return "None found"
    return "Invalid result"

# Use in your code