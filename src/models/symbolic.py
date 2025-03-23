import numpy as np
import sympy as sp
from sympy import symbols, simplify, expand, Function
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union, Callable
import random
import copy
import time
import os
import json
from tqdm import tqdm

# Try to import gplearn if available
try:
    from gplearn.genetic import SymbolicRegressor
    GPLEARN_AVAILABLE = True
except ImportError:
    GPLEARN_AVAILABLE = False
    print("Warning: gplearn not installed. Some symbolic regression features will be limited.")

class LagrangianSymbolicRegression:
    """
    Symbolic regression for discovering new Lagrangians or completing partial ones.
    Uses genetic programming to evolve mathematical expressions.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize symbolic regression with configuration
        
        Args:
            config: Dictionary with configuration parameters
        """
        self.config = {
            'population_size': 500,
            'generations': 20,
            'tournament_size': 20,
            'p_crossover': 0.7,
            'p_mutation': 0.1,
            'max_tree_depth': 5,
            'metric': 'mean_squared_error',
            'parsimony_coefficient': 0.1,  # Penalty for complexity
            'random_state': 42,
            'n_jobs': -1  # Use all available cores
        }
        
        if config:
            self.config.update(config)
        
        # Define physical variables that commonly appear in Lagrangians
        self.common_variables = {
            'position': symbols('x y z r theta phi'),
            'velocity': symbols('v v_x v_y v_z r_dot theta_dot phi_dot'),
            'fields': symbols('phi A B E psi'),
            'constants': symbols('m g k q c hbar G epsilon_0 mu_0'),
            'time': symbols('t')
        }
        
        # Common operators in Lagrangians
        self.operators = ['+', '-', '*', '/', '**', 'sqrt', 'exp', 'sin', 'cos']
        
        # Physical principles that Lagrangians often satisfy
        self.physical_principles = [
            'conservation_of_energy',
            'conservation_of_momentum',
            'gauge_invariance',
            'locality',
            'lorentz_invariance'
        ]
        
        # Initialize results storage
        self.best_individuals = []
    
    def _init_variable_space(self, domain: str) -> List[sp.Symbol]:
        """Initialize variable space based on physics domain"""
        if domain == 'classical_mechanics':
            return list(self.common_variables['position']) + list(self.common_variables['velocity']) + [self.common_variables['constants'][0]]  # m for mass
        
        elif domain == 'electrodynamics':
            return list(self.common_variables['fields'][:3]) + [self.common_variables['constants'][3]]  # q for charge
        
        elif domain == 'quantum_mechanics':
            return [self.common_variables['fields'][4]] + [self.common_variables['constants'][5]]  # psi, hbar
        
        else:
            # Generic set of variables
            return [var for var_type in self.common_variables.values() for var in var_type][:10]
    
    def define_search_space(self, domain: str, constraints: List[str] = None) -> Dict:
        """
        Define the search space for symbolic regression based on physics domain
        
        Args:
            domain: Physics domain (e.g., 'classical_mechanics', 'electrodynamics')
            constraints: List of physical principles to enforce
            
        Returns:
            Dictionary with search space parameters
        """
        variables = self._init_variable_space(domain)
        
        # Adjust operators based on domain
        if domain == 'classical_mechanics':
            operators = ['+', '-', '*', '/', '**', 'sqrt']
        elif domain == 'quantum_mechanics':
            operators = ['+', '-', '*', '/', '**', 'sqrt', 'exp', 'sin', 'cos']
        else:
            operators = self.operators
        
        # Apply constraints
        if constraints:
            # Implement constraint logic here
            pass
        
        return {
            'variables': variables,
            'operators': operators,
            'max_terms': 5 if domain == 'classical_mechanics' else 10,
            'domain': domain
        }
    
    def _fitness_function(self, expr, target_data=None, principles=None):
        """Enhanced fitness function with physics knowledge"""
        # Initialize with complexity penalty
        fitness = len(str(expr)) * self.config['parsimony_coefficient']
        
        # Reject overly complex expressions
        if len(str(expr)) > 300:
            return -float('inf')
        
        try:
            expr_str = str(expr)
            physics_score = 0
            
            # Check for kinetic energy terms (usually positive, with squared velocity)
            has_kinetic_term = False
            for v_term in ['v**2', 'v_x**2', 'v_y**2', 'v_z**2', 'r_dot**2', 'theta_dot**2', 'phi_dot**2']:
                if v_term in expr_str:
                    has_kinetic_term = True
                    physics_score += 5
            
            # Check for potential energy terms (usually negative, with position)
            has_potential_term = False
            for p_term in ['x**2', 'y**2', 'z**2', 'r**2', 'r']:
                if p_term in expr_str and '-' in expr_str:
                    has_potential_term = True
                    physics_score += 5
            
            # T-V structure is typical for Lagrangians
            if has_kinetic_term and has_potential_term:
                physics_score += 10
            
            # Apply physical principles
            principles_score = 0
            if principles:
                for principle in principles:
                    if principle == 'conservation_of_energy':
                        # For energy conservation, prefer time-independent expressions
                        vars_list = [str(s) for s in expr.free_symbols]
                        if 't' not in vars_list:
                            principles_score += 15
                    
                    elif principle == 'gauge_invariance':
                        # Simplified check for gauge invariance
                        principles_score += 10
            
            # Final score (higher is better)
            total_score = physics_score + principles_score - fitness
            return total_score
        except Exception:
            return -float('inf')
    
    def generate_candidate_lagrangian(self, search_space: Dict) -> sp.Expr:
        """Generate general physics-informed Lagrangian candidates"""
        variables = search_space.get('variables', [])
        if not variables:
            variables = [sp.Symbol(var) for var in ['m', 'v', 'x']]
        
        max_terms = search_space.get('max_terms', 3)
        domain = search_space.get('domain', 'generic')
        
        # Define general structures that appear across physics domains
        structures = [
            'kinetic',        # Terms with squared velocities
            'potential',      # Terms with positions
            'interaction',    # Cross-terms between variables
            'field',          # Field terms
            'random'          # Completely random terms
        ]
        
        # Build expression as sum of different term types
        expr = 0
        term_count = random.randint(1, max_terms)
        
        for _ in range(term_count):
            # Choose a structure with physics-informed probabilities
            structure = random.choices(
                structures, 
                weights=[0.3, 0.3, 0.2, 0.1, 0.1],  # Favor kinetic and potential
                k=1
            )[0]
            
            if structure == 'kinetic':
                # Create kinetic-like term (squared variable, usually positive)
                velocity_vars = [v for v in variables if 'v' in str(v) or 'dot' in str(v)]
                if velocity_vars:
                    var = random.choice(velocity_vars)
                    coef = random.uniform(0.1, 2.0)
                    expr += coef * var**2
                else:
                    # Fall back to any variable
                    var = random.choice(variables)
                    expr += random.uniform(0.1, 2.0) * var**2
                    
            elif structure == 'potential':
                # Create potential-like term (usually negative)
                position_vars = [v for v in variables if not ('v' in str(v) or 'dot' in str(v))]
                if position_vars:
                    var = random.choice(position_vars)
                    coef = random.uniform(0.1, 2.0)
                    if random.random() < 0.7:  # Usually negative
                        expr -= coef * var**random.choice([1, 2])
                    else:
                        expr += coef * var**random.choice([1, 2])
                else:
                    # Fall back to any variable
                    var = random.choice(variables)
                    expr -= random.uniform(0.1, 2.0) * var**random.choice([1, 2])
                    
            elif structure == 'interaction':
                # Create interaction terms (products of variables)
                if len(variables) >= 2:
                    var1, var2 = random.sample(variables, 2)
                    coef = random.uniform(-2.0, 2.0)
                    expr += coef * var1 * var2
                else:
                    # Not enough variables for interaction
                    var = random.choice(variables)
                    expr += random.uniform(-2.0, 2.0) * var
                    
            elif structure == 'field':
                # Create field-like terms
                field_vars = [v for v in variables if str(v) in ['A', 'B', 'E', 'phi']]
                if field_vars and len(variables) >= 2:
                    field = random.choice(field_vars)
                    var = random.choice([v for v in variables if v != field])
                    coef = random.uniform(-2.0, 2.0)
                    expr += coef * field * var
                else:
                    # Not enough field variables
                    var = random.choice(variables)
                    expr += random.uniform(-2.0, 2.0) * var
                    
            else:  # random structure
                # Create totally random term
                var = random.choice(variables)
                coef = random.uniform(-5.0, 5.0)
                power = random.choice([1, 2, 0.5])
                expr += coef * var**power
        
        # Apply simplification
        return sp.simplify(expr)

    
    def _generate_random_expression(self, search_space: Dict) -> sp.Expr:
        """Generate a random expression with physics-informed structure"""
        variables = search_space.get('variables', [])
        if not variables:
            variables = [sp.Symbol(var) for var in ['m', 'v', 'x']]
        
        max_terms = search_space.get('max_terms', 3)
        
        # Generate a simpler expression with fewer terms
        expr = 0
        term_count = random.randint(1, min(max_terms, 3))  # Limit to 3 terms max
        
        for _ in range(term_count):
            # Choose term type (quadratic, linear, interaction)
            term_type = random.choice(['quadratic', 'linear', 'interaction'])
            
            if term_type == 'quadratic':
                # Generate a term like a*v^2
                var = random.choice(variables)
                coef = random.uniform(0.1, 2.0)
                expr += coef * var**2
                
            elif term_type == 'linear':
                # Generate a term like b*x
                var = random.choice(variables)
                coef = random.uniform(-2.0, 2.0)
                expr += coef * var
                
            else:  # interaction
                # Generate a product term like c*v*x
                if len(variables) >= 2:
                    var1, var2 = random.sample(variables, 2)
                    coef = random.uniform(-2.0, 2.0)
                    expr += coef * var1 * var2
                else:
                    var = random.choice(variables)
                    expr += random.uniform(-2.0, 2.0) * var
        
        return expr

    def _mutate(self, expr, search_space: Dict) -> sp.Expr:
        """Mutate a symbolic expression"""
        # Convert to string representation
        expr_str = str(expr)
        
        # Define possible mutation operations
        mutations = [
            'replace_term',
            'add_term',
            'remove_term',
            'replace_operator',
            'replace_variable'
        ]
        
        # Select a mutation operation
        mutation = random.choice(mutations)
        
        if mutation == 'replace_term':
            # Generate a new term and use it to replace a random part
            new_expr = self.generate_candidate_lagrangian(search_space)
            return new_expr
        
        elif mutation == 'add_term':
            # Add a new term
            new_term = self.generate_candidate_lagrangian({**search_space, 'max_terms': 1})
            if random.random() < 0.5:
                return expr + new_term
            else:
                return expr - new_term
        
        elif mutation == 'remove_term':
            # Try to remove a term if expression is a sum
            if isinstance(expr, sp.Add) and len(expr.args) > 1:
                args = list(expr.args)
                args.pop(random.randrange(len(args)))
                return sum(args)
            else:
                # If not a sum or only one term, return original
                return expr
        
        elif mutation == 'replace_operator':
            # This is complex for symbolic expressions
            # For simplicity, generate a new expression
            return self.generate_candidate_lagrangian(search_space)
        
        elif mutation == 'replace_variable':
            # Replace a variable with another
            variables = search_space['variables']
            if variables and isinstance(expr, sp.Symbol) and expr in variables:
                new_var = random.choice(variables)
                return expr.subs(expr, new_var)
            return expr
        
        # Default: return original
        return expr
    
    def _crossover(self, expr1, expr2):
        """Perform crossover between two symbolic expressions"""
        # Simple implementation: randomly combine parts of expressions
        if random.random() < 0.5 and isinstance(expr1, sp.Add) and isinstance(expr2, sp.Add):
            # Take terms from both expressions
            terms1 = list(expr1.args)
            terms2 = list(expr2.args)
            
            # Randomly select terms from each
            n_terms1 = random.randint(1, max(1, len(terms1)))
            n_terms2 = random.randint(1, max(1, len(terms2)))
            
            selected_terms1 = random.sample(terms1, min(n_terms1, len(terms1)))
            selected_terms2 = random.sample(terms2, min(n_terms2, len(terms2)))
            
            # Combine terms
            return sum(selected_terms1 + selected_terms2)
        
        # Default: randomly choose between expressions
        return expr1 if random.random() < 0.5 else expr2
    
    def evolve_lagrangians(
        self, 
        search_space: Dict, 
        fitness_function: Callable = None, 
        target_data: Dict = None,
        principles: List[str] = None,
        early_stopping: int = 5,
        output_dir: str = None
    ) -> List[sp.Expr]:
        """
        Evolve a population of Lagrangians using genetic programming
        
        Args:
            search_space: Dictionary defining the search space
            fitness_function: Custom fitness function (if None, use internal function)
            target_data: Data to fit expressions to (optional)
            principles: List of physical principles to enforce
            early_stopping: Stop if no improvement after this many generations
            output_dir: Directory to save results
            
        Returns:
            List of best Lagrangian expressions found
        """
        # If gplearn is available and we have target data, use it
        if GPLEARN_AVAILABLE and target_data is not None:
            return self._evolve_with_gplearn(search_space, target_data, output_dir)
        
        # Use internal implementation
        return self._evolve_internal(search_space, fitness_function, target_data, principles, early_stopping, output_dir)
    
    def _evolve_internal(
        self,
        search_space: Dict,
        fitness_function: Callable = None,
        target_data: Dict = None,
        principles: List[str] = None,
        early_stopping: int = 5,
        output_dir: str = None
    ) -> List[sp.Expr]:
        """Internal implementation of evolutionary algorithm"""
        # Use provided fitness function or internal one
        fitness_func = fitness_function if fitness_function else lambda expr: self._fitness_function(expr, target_data, principles)
        
        # Initialize population
        population_size = self.config['population_size']
        population = [self.generate_candidate_lagrangian(search_space) for _ in range(population_size)]
        
        # Add the domain-aware seeding
        seed_population = []
        
        # Add physics-informed seed candidates
        domain = search_space.get('domain', 'generic')
        if domain == 'electromagnetic':
            # Add EM-like candidates (only if we have the right variables)
            var_names = [str(v) for v in search_space.get('variables', [])]
            if all(v in var_names for v in ['m', 'v', 'q', 'A']):
                m, v, q, A = [next(var for var in search_space['variables'] if str(var) == sym)
                               for sym in ['m', 'v', 'q', 'A']]
                seed_population.append(0.5 * m * v**2 + q * A * v)
        
        # Evaluate initial population
        fitness_scores = [fitness_func(ind) for ind in population]
        
        # Sort by fitness (higher is better)
        sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: -pair[0])]
        best_fitness = max(fitness_scores)
        
        # Track best individual
        best_individual = sorted_population[0]
        self.best_individuals = [best_individual]
        
        # Generation loop
        generations = self.config['generations']
        no_improvement = 0
        
        for generation in tqdm(range(generations), desc="Evolving Lagrangians"):
            # Create new generation
            new_population = []
            
            # Elitism: keep best individuals
            elite_size = max(1, int(0.05 * population_size))
            new_population.extend(sorted_population[:elite_size])
            
            # Fill the rest with crossover and mutation
            while len(new_population) < population_size:
                # Tournament selection
                tournament_size = min(self.config['tournament_size'], len(sorted_population))
                candidates = random.sample(sorted_population, tournament_size)
                parent1 = candidates[0]  # Already sorted by fitness
                
                if random.random() < self.config['p_crossover'] and len(sorted_population) > 1:
                    # Crossover
                    candidates = random.sample(sorted_population, tournament_size)
                    parent2 = candidates[0]
                    offspring = self._crossover(parent1, parent2)
                else:
                    offspring = copy.copy(parent1)
                
                # Mutation
                if random.random() < self.config['p_mutation']:
                    offspring = self._mutate(offspring, search_space)
                
                # Add to new population
                new_population.append(offspring)
            
            # Replace old population
            population = new_population
            
            # Evaluate new population
            fitness_scores = [fitness_func(ind) for ind in population]
            
            # Sort by fitness
            sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: -pair[0])]
            generation_best_fitness = max(fitness_scores)
            
            # Check if we have a new best
            if generation_best_fitness > best_fitness:
                best_fitness = generation_best_fitness
                best_individual = sorted_population[0]
                self.best_individuals.append(best_individual)
                no_improvement = 0
            else:
                no_improvement += 1
            
            # Early stopping
            if no_improvement >= early_stopping:
                print(f"Stopping early after {generation+1} generations without improvement")
                break
            
            # Print progress
            if (generation + 1) % 5 == 0 or generation == 0:
                simplified = sp.simplify(best_individual)
                print(f"Generation {generation+1}: Best fitness = {best_fitness}")
                print(f"Best Lagrangian: L = {simplified}")
        
        # Save results
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            results = {
                "best_fitness": best_fitness,
                "best_lagrangian": str(sp.simplify(best_individual)),
                "all_best_lagrangians": [str(sp.simplify(ind)) for ind in self.best_individuals],
                "search_space": {k: str(v) if not isinstance(v, list) else [str(item) for item in v] 
                                for k, v in search_space.items()}
            }
            
            with open(os.path.join(output_dir, "symbolic_regression_results.json"), 'w') as f:
                json.dump(results, f, indent=2)
        
        # Return top individuals, simplified
        return [sp.simplify(ind) for ind in self.best_individuals]
    
    def _evolve_with_gplearn(self, search_space: Dict, target_data: Dict, output_dir: str = None) -> List[sp.Expr]:
        """Use gplearn for symbolic regression with target data"""
        if not GPLEARN_AVAILABLE:
            print("gplearn is not installed. Using internal implementation instead.")
            return self._evolve_internal(search_space, None, target_data, None, 5, output_dir)
        
        # Extract X and y from target data
        X = np.array(target_data['X'])
        y = np.array(target_data['y'])
        
        # Configure gplearn
        est_gp = SymbolicRegressor(
            population_size=self.config['population_size'],
            generations=self.config['generations'],
            tournament_size=self.config['tournament_size'],
            p_crossover=self.config['p_crossover'],
            p_subtree_mutation=self.config['p_mutation'],
            p_hoist_mutation=self.config['p_mutation'] / 2,
            p_point_mutation=self.config['p_mutation'] / 2,
            max_samples=0.9,
            parsimony_coefficient=self.config['parsimony_coefficient'],
            random_state=self.config['random_state'],
            n_jobs=self.config['n_jobs']
        )
        
        # Fit model
        est_gp.fit(X, y)
        
        # Extract best programs
        best_programs = [est_gp._program]
        
        # Convert to SymPy expressions
        sympy_expressions = []
        var_names = [f"x{i}" for i in range(X.shape[1])]
        vars_sympy = symbols(' '.join(var_names))
        
        for program in best_programs:
            # Convert gplearn program to SymPy expression
            # (Implementation depends on gplearn's structure)
            expr_str = str(program)
            try:
                # Replace variable names
                for i, var in enumerate(var_names):
                    expr_str = expr_str.replace(f"X{i}", var)
                
                # Parse expression
                expr = sp.sympify(expr_str)
                sympy_expressions.append(expr)
            except:
                print(f"Failed to convert expression: {expr_str}")
        
        # Save results
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            results = {
                "best_fitness": -est_gp._program.raw_fitness_,
                "best_lagrangian": str(sympy_expressions[0]) if sympy_expressions else "Conversion failed",
                "gplearn_expression": str(est_gp._program)
            }
            
            with open(os.path.join(output_dir, "gplearn_results.json"), 'w') as f:
                json.dump(results, f, indent=2)
        
        return sympy_expressions if sympy_expressions else [None]
    
    def lagrangian_from_equations_of_motion(
        self, 
        equations_of_motion: List[str],
        variables: List[str],
        output_dir: str = None
    ) -> sp.Expr:
        """
        Derive a Lagrangian from equations of motion
        
        This is a special case of symbolic regression where we know the
        resulting Euler-Lagrange equations should match our input equations.
        
        Args:
            equations_of_motion: List of equation strings
            variables: List of variable names
            output_dir: Directory to save results
            
        Returns:
            Sympy expression for the derived Lagrangian
        """
        # Parse equations of motion
        parsed_eqs = []
        for eq_str in equations_of_motion:
            try:
                parsed_eqs.append(sp.sympify(eq_str))
            except:
                print(f"Failed to parse equation: {eq_str}")
                return None
        
        # Create symbolic variables
        sym_vars = symbols(' '.join(variables))
        if not isinstance(sym_vars, tuple):
            sym_vars = (sym_vars,)
        
        # Create placeholder for the Lagrangian
        L = sp.Function('L')(*sym_vars)
        
        # TODO: Implement the inverse variational problem
        # This is an advanced mathematical problem beyond the scope
        # of a simple implementation. For now, we'll use our genetic
        # algorithm approach to find a Lagrangian that approximately
        # satisfies the equations of motion.
        
        print("Note: Deriving exact Lagrangians from equations of motion is an advanced problem.")
        print("Using symbolic regression to find an approximate Lagrangian.")
        
        # Create a search space with our variables
        search_space = {
            'variables': sym_vars,
            'operators': self.operators,
            'max_terms': 10,
            'domain': 'generic'
        }
        
        # Define a fitness function that measures how well the derived 
        # Euler-Lagrange equations match our input equations
        def fitness_from_eom(expr):
            # Convert expr to a sympy expression if it's not already
            if not isinstance(expr, sp.Expr):
                try:
                    expr = sp.sympify(expr)
                except:
                    return -float('inf')  # Extremely bad fitness
            
            # Compute Euler-Lagrange equations for this Lagrangian
            derived_eqs = []
            for var in sym_vars:
                # This is a simplified approach - real implementation would be more complex
                # d/dt(∂L/∂ẋ) - ∂L/∂x = 0
                # For simplicity, we're skipping the time derivative part
                try:
                    eq = sp.diff(expr, var)
                    derived_eqs.append(eq)
                except:
                    return -float('inf')  # Extremely bad fitness
            
            # Compare derived equations with input equations
            total_diff = 0
            for i, (derived, target) in enumerate(zip(derived_eqs, parsed_eqs)):
                # Simplified: compare string representations
                # Real implementation would use more sophisticated equation comparison
                diff = sum(1 for a, b in zip(str(derived), str(target)) if a != b)
                total_diff += diff
            
            # Add complexity penalty
            complexity = len(str(expr)) * self.config['parsimony_coefficient']
            
            # Convert to fitness (negative because lower diff is better)
            return -(total_diff + complexity)
        
        # Use our genetic algorithm to evolve a Lagrangian
        best_lagrangians = self._evolve_internal(
            search_space,
            fitness_function=fitness_from_eom,
            early_stopping=10,
            output_dir=output_dir
        )
        
        # Return the best Lagrangian found
        return best_lagrangians[0] if best_lagrangians else None
    
    def discover_new_lagrangians(
        self,
        domain: str,
        constraints: List[str] = None,
        output_dir: str = "results/symbolic_regression"
    ) -> List[sp.Expr]:
        """
        Discover new potential Lagrangians for a given physics domain
        
        Args:
            domain: Physics domain (e.g., 'classical_mechanics')
            constraints: Physical constraints to enforce
            output_dir: Directory to save results
            
        Returns:
            List of discovered Lagrangian expressions
        """
        print(f"Discovering new Lagrangians for domain: {domain}")
        
        # Define search space based on domain
        search_space = self.define_search_space(domain, constraints)
        
        # Select physical principles relevant to the domain
        if domain == 'classical_mechanics':
            principles = ['conservation_of_energy', 'conservation_of_momentum']
        elif domain == 'electrodynamics':
            principles = ['gauge_invariance', 'lorentz_invariance']
        else:
            principles = self.physical_principles
        
        # Evolve Lagrangians with these principles
        lagrangians = self.evolve_lagrangians(
            search_space,
            principles=principles,
            output_dir=os.path.join(output_dir, domain)
        )
        
        # Print results
        print(f"\nDiscovered {len(lagrangians)} potential Lagrangians:")
        for i, lagr in enumerate(lagrangians):
            print(f"{i+1}. L = {lagr}")
        
        return lagrangians

    @staticmethod
    def _check_electromagnetic(expr: sp.Expr) -> Dict:
        """Check electromagnetic field Lagrangian properties"""
        result = {"properties": [], "score": 0, "suggestions": []}
        
        # Get variables and expression as string
        variables = [str(s) for s in expr.free_symbols]
        expr_str = str(expr)
        
        # Expand the expression to identify terms in any form
        expanded_expr = sp.expand(expr)
        expanded_str = str(expanded_expr)
        
        # Check for required variables
        has_q = 'q' in variables
        has_A = 'A' in variables
        has_v = 'v' in variables
        has_m = 'm' in variables
        
        # Add properties for present variables
        if has_q:
            result["properties"].append("has_charge")
            result["score"] += 10
        else:
            result["suggestions"].append("Missing charge variable 'q'")
        
        if has_A:
            result["properties"].append("has_vector_potential")
            result["score"] += 10
        else:
            result["suggestions"].append("Missing vector potential 'A'")
        
        # Check for interactions in both original and expanded forms
        interactions = ['q*A*v', 'A*q*v', 'q*v*A', 'v*q*A', 'v*A*q', 'A*v*q']
        has_interaction = has_q and has_A and has_v and (
            any(term in expr_str for term in interactions) or
            any(term in expanded_str for term in interactions)
        )
        
        # Check for kinetic terms in both original and expanded forms
        kinetic_terms = ['m*v**2', '0.5*m*v**2', 'm*v^2', '(1/2)*m*v^2']
        has_kinetic = has_m and has_v and (
            any(term in expr_str for term in kinetic_terms) or  
            any(term in expanded_str for term in kinetic_terms) or
            '0.5*m*v**2' in expanded_str or 'm*v**2' in expanded_str
        )
        
        if has_kinetic:
            result["properties"].append("has_kinetic_energy")
            result["score"] += 15
        else:
            result["suggestions"].append("Missing kinetic energy term (1/2)mv²")
            
        if has_interaction:
            result["properties"].append("has_em_interaction")
            result["score"] += 25
        else:
            result["suggestions"].append("Missing electromagnetic interaction term q(A·v)")
        
        return result

# Example usage function
def discover_gravitational_lagrangians():
    """Example: Discover new Lagrangians for gravitational systems"""
    symbolic_engine = LagrangianSymbolicRegression()
    
    # Set parameters for gravitational systems
    config = {
        'population_size': 200,
        'generations': 10,
        'parsimony_coefficient': 0.05  # Prefer simpler expressions
    }
    symbolic_engine.config.update(config)
    
    # Discover new Lagrangians
    lagrangians = symbolic_engine.discover_new_lagrangians(
        'classical_mechanics',
        constraints=['conservation_of_energy'],
        output_dir="results/gravitational_lagrangians"
    )
    
    return lagrangians