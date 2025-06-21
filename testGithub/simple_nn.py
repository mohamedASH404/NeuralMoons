import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--model_path', type=str, default='best_model.pth')
parser.add_argument('--load_model', action='store_true')
args = parser.parse_args()

# Data
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.LongTensor(y_train).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.LongTensor(y_test).to(device)

# Model
class ImprovedNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.model(x)

model = ImprovedNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

def train(model, optimizer, criterion, X, y, X_val, y_val, epochs=200, patience=20):
    best_loss = float('inf')
    wait = 0
    losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_loss = criterion(val_out, y_val)

        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            wait = 0
            best_model_state = model.state_dict()
            torch.save(best_model_state, args.model_path)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

    # Plot training loss
    plt.plot(losses, label='Training Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.show()

# Train or Load
if args.load_model and os.path.exists(args.model_path):
    print("Loading saved model...")
    model.load_state_dict(torch.load(args.model_path))
else:
    print("Training model...")
    train(model, optimizer, criterion, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor,
          epochs=args.epochs, patience=args.patience)

# Evaluation
model.eval()
with torch.no_grad():
    out = model(X_test_tensor)
    pred = torch.argmax(out, 1)
    acc = (pred == y_test_tensor).float().mean()
    print(f"Test Accuracy: {acc:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, pred.cpu().numpy()))

    # Confusion Matrix
    cm = confusion_matrix(y_test, pred.cpu().numpy())
    print("Confusion Matrix:\n", cm)

# Plot decision boundary
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(device)
    with torch.no_grad():
        Z = model(grid)
        Z = torch.argmax(Z, axis=1).cpu().numpy()
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolors='k')
    plt.title("Improved Decision Boundary")
    plt.show()

plot_decision_boundary(model, X_test, y_test)
