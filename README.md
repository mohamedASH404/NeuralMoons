# Neural Network on Moons Dataset

This project trains a simple neural network on the "moons" dataset using PyTorch.

## Requirements

Install dependencies with:
```bash
pip install -r requirements.txt
```

Dependencies: torch, scikit-learn, matplotlib

## How to run

Run the script with:
```bash
python simple_nn.py
```

It trains the model for 100 epochs, prints loss every 10 epochs, then shows accuracy and decision boundary plot.

## Code explanation

- Generate 2D moons dataset with noise
- Scale features and split train/test
- Define a small neural network (2 input -> 16 hidden -> 2 output)
- Use cross-entropy loss and Adam optimizer.
- Train for 100 epochs
- Evaluate accuracy on test set.
- Plot decision boundary.

The code uses PyTorch for modeling and training.
