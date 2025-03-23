# Neural Lagrangian Generator (Mini Version)

## Overview
The Neural Lagrangian Generator is a project aimed at training a small transformer model to predict terms in a Lagrangian given a dataset of known physics equations. The goal is to explore whether the model can learn patterns in fundamental physics laws and suggest meaningful Lagrangian structures.

## Features
- **Dataset Handling**: Load and preprocess datasets of physics equations.
- **Tokenization**: Convert equations into a format suitable for model training using SymPy and LaTeX-style tokenization.
- **Model Architecture**: Implement a transformer-based model for predicting Lagrangian terms.
- **Training Loop**: Manage the training process, including optimization and checkpointing.
- **Evaluation**: Assess model performance and visualize results.
- **CUDA Optimization**: Leverage GPU acceleration for faster training.

## Installation
To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd neural-lagrangian-generator
pip install -r requirements.txt
```

## Usage

### Data Preprocessing
Preprocess the raw physics equations data:

```bash
python src/cli.py preprocess --input data/raw/physics_equations.json --output data/processed/tokenized_equations.json
```

### Training
Train the model with GPU acceleration:

```bash
python src/cli.py train --model gpt2 --data-dir data --output-dir models --batch-size 4 --epochs 5
```

To train on CPU only:

```bash
python src/cli.py train --model gpt2 --no-cuda
```

### Generation
Generate a Lagrangian with a prompt:

```bash
python src/cli.py generate --model-path models/best_model --prompt "Lagrangian for a charged particle in an electromagnetic field"
```

### Evaluation
Evaluate the model's performance:

```bash
python src/cli.py evaluate --model-path models/best_model --test-data data/processed/training_examples.json
```

## Project Structure
```
neural-lagrangian-generator/
├── data/
│   ├── raw/
│   │   └── physics_equations.json
│   └── processed/
│       ├── tokenized_equations.json
│       └── training_examples.json
├── models/
│   └── best_model/
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   └── tokenizer.py
│   ├── models/
│   │   └── transformer.py
│   ├── training/
│   │   └── train.py
│   ├── evaluation/
│   │   └── metrics.py
│   └── cli.py
├── configs/
│   └── model_config.yaml
├── notebooks/
│   ├── data_exploration.ipynb
│   └── model_evaluation.ipynb
├── results/
├── requirements.txt
├── setup.py
└── README.md
```

## Notebooks
- **Data Exploration**: `notebooks/data_exploration.ipynb` - Explore the dataset and visualize equations
- **Model Evaluation**: `notebooks/model_evaluation.ipynb` - Analyze the trained model's performance

## Configuration
Model hyperparameters and architecture details can be adjusted in the `configs/model_config.yaml` file.

## License
This project is licensed under the MIT License.

## Acknowledgments
- Inspired by fundamental physics principles and the potential of machine learning in scientific discovery.