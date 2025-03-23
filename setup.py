from setuptools import setup, find_packages

setup(
    name='neural-lagrangian-generator',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A project to generate Lagrangians using a neural network model.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch>=1.7.0',
        'transformers>=4.0.0',
        'sympy>=1.7.0',
        'numpy>=1.19.0',
        'pandas>=1.1.0',
        'matplotlib>=3.3.0',
        'scikit-learn>=0.24.0',
        'PyYAML>=5.3.0',
    ],
    python_requires='>=3.6',
)