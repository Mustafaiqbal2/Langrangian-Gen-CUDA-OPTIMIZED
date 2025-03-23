import matplotlib.pyplot as plt
import numpy as np

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