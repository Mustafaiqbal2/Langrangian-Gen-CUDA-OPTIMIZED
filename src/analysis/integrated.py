import os
import matplotlib.pyplot as plt
import sympy as sp
from typing import Dict, Any
from src.evaluation.physics_test import PhysicsEvaluator

def analyze_physics_system(context, model, tokenizer, visualizer, symbolic_engine, temp=0.3):
    """
    Integrated analysis function combining transformer generation, 
    visualization, and symbolic regression with error handling.
    """
    results = {
        "success": True,
        "errors": [],
        "generated_lagrangian": None,
        "tokens": None,
        "analysis": None,
        "variations": None
    }
    
    # Step 1: Generate a Lagrangian using the transformer model
    try:
        generated = model.generate_lagrangian(context, temperature=temp)[0]
        results["generated_lagrangian"] = generated
        print("Generated Lagrangian:")
        print("-" * 40)
        print(generated)
        print("-" * 40)
    except Exception as e:
        error_msg = f"Error generating Lagrangian: {str(e)}"
        results["errors"].append(error_msg)
        results["success"] = False
        print(error_msg)
        return results
    
    # Step 2: Analyze and tokenize the generated equation
    equation_part = generated
    if "L = " in generated:
        equation_part = generated.split("L = ")[-1]
    
    try:
        tokens = tokenizer.tokenize_equation(equation_part)
        results["tokens"] = tokens
        print("\nTokenized equation:")
        print(tokens)
    except Exception as e:
        error_msg = f"Error tokenizing equation: {str(e)}"
        results["errors"].append(error_msg)
        print(error_msg)
    
    try:
        analysis = tokenizer.analyze_equation(equation_part)
        results["analysis"] = analysis
        
        if analysis["valid_parse"]:
            print("\nEquation analysis successful:")
            print(f"Symbolic expression: {analysis['symbolic_expr']}")
            print(f"Symbols: {', '.join(str(s) for s in analysis['symbols'])}")
            print(f"Term count: {analysis['term_count']}")
        else:
            print("\nEquation parsing failed:")
            print(f"Error: {analysis['error']}")
            print(f"Using tokens as fallback: {analysis.get('potential_symbols', [])}")
    except Exception as e:
        error_msg = f"Error analyzing equation: {str(e)}"
        results["errors"].append(error_msg)
        print(error_msg)
    
    # Step 3: Visualize attention patterns
    try:
        prompt = f"Given the following physics context, generate the corresponding Lagrangian equation:\nContext: {context}\nLagrangian equation: L = "
        fig = visualizer.plot_attention_map(prompt, layer=-1, head=0, show_plot=True)
        print("\nAttention visualization complete")
    except Exception as e:
        error_msg = f"Error visualizing attention: {str(e)}"
        results["errors"].append(error_msg)
        print(error_msg)
    
    # Step 4: Use symbolic regression to find variations
    try:
        # Get symbols for search space
        if results["analysis"] and results["analysis"]["valid_parse"]:
            # Use parsed symbols
            symbols = list(results["analysis"]["symbols"])
        elif results["analysis"] and "potential_symbols" in results["analysis"]:
            # Use potential symbols from tokenization
            symbols = [sp.Symbol(s) for s in results["analysis"]["potential_symbols"]]
        else:
            # Fallback to basic physics variables
            symbols = [sp.Symbol(s) for s in ['m', 'x', 'v', 'k', 'g']]
        
        # Create search space for symbolic regression
        custom_search_space = {
            'variables': symbols,
            'operators': symbolic_engine.operators[:4],  # Use simpler operators
            'max_terms': 4,
            'domain': 'classical_mechanics'  # Assume classical mechanics as default
        }
        
        # Find variations
        print("\nFinding variations of the Lagrangian...")
        
        # Update config for faster processing
        symbolic_engine.config.update({
            'population_size': 100,
            'generations': 5,
            'parsimony_coefficient': 0.05
        })
        
        variations = symbolic_engine.evolve_lagrangians(
            custom_search_space,
            principles=['conservation_of_energy'],
            early_stopping=2,
            output_dir="../results/variations"
        )
        
        results["variations"] = [str(v) for v in variations]
        
        print("\nPossible variations:")
        for i, var in enumerate(variations):
            print(f"{i+1}. L = {var}")
            
    except Exception as e:
        error_msg = f"Error in symbolic regression: {str(e)}"
        results["errors"].append(error_msg)
        print(error_msg)
    
    # Add physics evaluation step
    try:
        # Determine domain based on context
        domain = "classical"
        if any(term in context.lower() for term in ["charge", "electric", "magnetic", "field"]):
            domain = "electromagnetic"
        elif any(term in context.lower() for term in ["quantum", "wave", "particle"]):
            domain = "quantum"
        
        print(f"\nEvaluating against physics principles (domain: {domain})")
        
        # Evaluate the generated Lagrangian if parsing was successful
        if results["analysis"] and results["analysis"].get("valid_parse", False):
            physics_eval = PhysicsEvaluator.check_lagrangian(
                results["analysis"]["symbolic_expr"], 
                domain=domain
            )
            results["physics_eval"] = physics_eval
            
            print(f"Physics score: {physics_eval['score']}/70")
            if physics_eval["properties"]:
                print("Properties satisfied:")
                for prop in physics_eval["properties"]:
                    print(f"- {prop}")
            
            if physics_eval["suggestions"]:
                print("\nSuggestions for improvement:")
                for suggestion in physics_eval["suggestions"]:
                    print(f"- {suggestion}")
        
        # Also evaluate the variations
        if results.get("variations"):
            print("\nEvaluating symbolic regression variations:")
            best_variation = None
            best_score = -1
            
            for i, var_str in enumerate(results["variations"]):
                try:
                    var_eval = PhysicsEvaluator.check_lagrangian(var_str, domain=domain)
                    print(f"Variation {i+1} physics score: {var_eval['score']}/70")
                    
                    if var_eval["score"] > best_score:
                        best_score = var_eval["score"]
                        best_variation = (i, var_str, var_eval)
                except Exception as e:
                    print(f"Error evaluating variation {i+1}: {e}")
            
            if best_variation:
                print(f"\nBest variation: #{best_variation[0]+1} with score {best_variation[2]['score']}/70")
                print(f"L = {best_variation[1]}")
                results["best_variation"] = best_variation[1]
                results["best_variation_score"] = best_variation[2]["score"]
    except Exception as e:
        error_msg = f"Error in physics evaluation: {str(e)}"
        results["errors"].append(error_msg)
        print(error_msg)
    
    return results