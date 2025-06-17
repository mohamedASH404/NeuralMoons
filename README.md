```markdown
# 🧠 Simple Neural Network with PyTorch

This project demonstrates a simple neural network built using PyTorch for binary classification on a synthetic dataset (`make_moons` from scikit-learn). It shows how a basic neural net can learn non-linear patterns through training.

---

##  How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/mohamedASH404/testGithub.git
cd testGithub
```

### 2. Install dependencies

Make sure you have Python 3.7+ installed. Then run:

```bash
pip install torch torchvision scikit-learn matplotlib
```

### 3. Run the model

```bash
python simple_nn.py
```

You will see training logs printed with the loss per epoch and the final test accuracy.

---

## 🧠 Model Architecture

- **Input**: 2 features (from `make_moons`)
- **Layers**:
  - `Linear(2 → 16)` + `ReLU`
  - `Linear(16 → 2)` (2 output classes)
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam
- **Epochs**: 100

---

## 📊 Results Summary

Here are several observed results across different runs:

| Run | Final Loss | Accuracy |
|-----|------------|----------|
| 1   | 0.2465     | 88.50%   |
| 2   | 0.2438     | 94.50%   |
| 3   | 0.1827     | 92.00%   |
| 4   | 0.2436     | 85.50%   |
| 5   | 0.1684     | 95.00%   |
| 6   | 0.1942     | 92.50%   |
| 7   | 0.2022     | 95.00%   |
| 8   | 0.2250     | 93.00%   |
| 9   | 0.2153     | 90.50%   |

- ✅ **Average Accuracy**: ~91.72%
- 📉 **Loss decreases steadily**, showing successful learning.
- 🔁 Accuracy fluctuates slightly due to randomness in data splits and weight initialization.
- ⚙️ Despite its simplicity, the model generalizes well.

---

## 📌 Notes

- You can improve performance by increasing the hidden layer size, changing the learning rate, or using additional layers.
- Adding a **decision boundary visualization** can help you understand how the model separates the data.
- The script is easily adaptable to other binary classification datasets.

---

## 👨‍💻 Author

**Mohamed Ait Sidi Hou**  
GitHub: [@mohamedASH404](https://github.com/mohamedASH404)
```
