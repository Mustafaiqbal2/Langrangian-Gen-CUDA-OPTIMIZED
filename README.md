# Lagrangian-Gen-CUDA-OPTIMIZED

A CUDA-optimized physics-based symbolic regression engine for discovering and generating Lagrangian equations from natural language descriptions. This project uses transformer-based language models combined with symbolic regression to generate physically meaningful mathematical expressions.

## ⚠️ PROJECT STATUS: INCOMPLETE ⚠️

**This project is currently in active development and is NOT fully functional.**

### Current Issues and Limitations:

1. **Physics Evaluation**: The classical mechanics evaluator (`_check_classical`) has implementation issues that cause test failures.
2. **Tokenization**: Basic equation parsing works correctly only for a subset of formats. Many equation formats fail to tokenize or parse properly.
3. **Symbolic Regression**: Works for electromagnetic domains but has issues with other domains.
4. **Transformer Integration**: The transformer model integration is incomplete and currently returns empty results during tests.
5. **CUDA Optimization**: While the code is structured for CUDA optimization, many components still run on CPU only.

## Project Structure

```
Langrangian-Gen-CUDA-OPTIMIZED/
├── src/
│   ├── evaluation/          # Physics evaluators
│   ├── tokenization/        # Equation parsing and tokenization
│   ├── symbolic/            # Symbolic regression engine
│   ├── models/              # Transformer models
│   ├── visualization/       # Result visualizers
│   └── integration/         # System integration
├── notebooks/               # Development notebooks
├── tests/                   # Test scripts
└── results/                 # Generated outputs
```

## Features (Planned)

- Natural language to physics equation conversion
- Physics-aware symbolic regression
- CUDA-accelerated computation
- Domain-specific Lagrangian evaluation (Classical, Electromagnetic, etc.)
- Interactive visualization of generated equations

## Requirements

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+ (for GPU acceleration)
- SymPy
- NumPy
- Matplotlib
- tqdm

## Installation

```bash
git clone https://github.com/yourusername/Langrangian-Gen-CUDA-OPTIMIZED.git
cd Langrangian-Gen-CUDA-OPTIMIZED
pip install -r requirements.txt
```

## Usage (When Completed)

```python
from src.integration.analyzer import analyze_physics_system

# Describe a physics system in natural language
context = "Lagrangian for a charged particle in a magnetic field"

# Analyze and generate Lagrangian
results = analyze_physics_system(context=context)

# Display generated Lagrangian
print(f"Generated Lagrangian: {results['generated_lagrangian']}")

# Explore alternative expressions
for i, expr in enumerate(results['symbolic_alternatives']):
    print(f"Alternative {i+1}: {expr}")
```

## Testing

To run validation tests:

```bash
python tests/physics_tester.py
```

The current test suite reveals several failing components that need to be addressed before the system is functional.

## Development Roadmap

1. Fix physics evaluator for classical mechanics domain ⚠️
2. Improve equation tokenization and parsing for all formats ⚠️
3. Complete the transformer model integration ⚠️ 
4. Optimize symbolic regression algorithm ⚠️
5. Add CUDA kernel support for large expression evaluation ⚠️
6. Create a comprehensive benchmark suite ⚠️
7. Develop a web interface for interactive exploration ⚠️

## Contributing

Contributions are welcome! This project is actively seeking contributors to help fix the current issues. Please see the issues tab for specific areas where help is needed.

---

**Note**: This README will be updated as the project progresses toward completion. The current focus is on fixing the core evaluation and tokenization components.