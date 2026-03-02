# 🧠 Simple Perceptron — From Scratch

A **single-layer perceptron** (linear regression model) built entirely from scratch using **PyTorch tensors and autograd** — no `nn.Module`, no `optim` package. Every component — forward pass, loss function, gradient descent, and the Adam optimiser — is hand-implemented to deeply understand how neural-network fundamentals work under the hood.

---

## 📌 Project Highlights

| Aspect | Details |
|---|---|
| **Model** | Single-layer Perceptron (Linear Regression) |
| **Framework** | PyTorch (tensors + autograd only) |
| **Optimisers** | Vanilla SGD **and** Adam (both hand-coded) |
| **Loss Function** | Root Mean Squared Error (RMSE) — custom implementation |
| **Dataset** | Student Performance dataset (`StudentPerformance.csv`, 10 000 rows) |
| **Train / Test Split** | Custom `train_test_split` function (no scikit-learn dependency for splitting) |
| **Data Loading** | PyTorch `DataLoader` with mini-batches of 32 |
| **Final R² Score** | **0.912** on the held-out test set |

---

## 🗂️ Repository Structure

```
├── Simple_perceptron.ipynb   # Main notebook — model, training & evaluation
├── StudentPerformance.csv    # Dataset (10 000 student records)
├── data.csv                  # Auxiliary data file
└── README.md                 # You are here
```

---

## 📊 Dataset

The **Student Performance** dataset contains **10 000 records** with the following features:

| Feature | Type | Description |
|---|---|---|
| `Hours Studied` | int | Daily hours spent studying |
| `Previous Scores` | int | Scores obtained in previous exams |
| `Extracurricular Activities` | str | Whether the student participates (`Yes` / `No`) — **dropped before training** |
| `Sleep Hours` | int | Average daily sleep hours |
| `Sample Question Papers Practiced` | int | Number of practice papers attempted |
| **`Performance Index`** | **float** | **Target variable** — overall performance score |

After dropping the categorical column (`Extracurricular Activities`), the model receives **4 numerical input features**.

---

## 🏗️ Architecture & Implementation Details

### Forward Pass

The model computes a simple linear transformation:

```
ŷ = X · W + b
```

where **W** ∈ ℝ⁴ (one weight per feature) and **b** ∈ ℝ¹ (bias scalar).

### Loss Function — RMSE (Custom)

```python
loss = torch.mean((y_true - y_pred) ** 2).sqrt()
```

Root Mean Squared Error is computed manually and is fully differentiable via PyTorch's autograd.

### Optimisers (Hand-Coded)

#### 1. Vanilla Gradient Descent

```
W ← W − η · ∂L/∂W
b ← b − η · ∂L/∂b
```

#### 2. Adam Optimiser ✅ *(used in training)*

Full implementation of the Adam algorithm with:

- **First moment estimate (m):** exponential moving average of gradients
- **Second moment estimate (v):** exponential moving average of squared gradients
- **Bias correction:** compensates for zero-initialised moments in early steps
- **Hyperparameters:** `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`, `lr = 0.001`

```
mₜ = β₁ · mₜ₋₁ + (1 − β₁) · gₜ
vₜ = β₂ · vₜ₋₁ + (1 − β₂) · gₜ²
m̂ₜ = mₜ / (1 − β₁ᵗ)
v̂ₜ = vₜ / (1 − β₂ᵗ)
θₜ = θₜ₋₁ − lr · m̂ₜ / (√v̂ₜ + ε)
```

### Custom Utilities

| Utility | Description |
|---|---|
| `train_test_split()` | Splits features & labels by a configurable ratio (default 75/25). Mimics scikit-learn's API. |
| `CustomDataset` | PyTorch `Dataset` subclass for integration with `DataLoader`. |
| `zero_grad()` | Manually resets gradients to `None` before each backward pass. |

---

## 🚀 Training Pipeline

```
                ┌──────────────────────────────┐
                │   Load CSV → NumPy Arrays    │
                └──────────────┬───────────────┘
                               ▼
                ┌──────────────────────────────┐
                │   Custom Train/Test Split    │
                │        (80 / 20)             │
                └──────────────┬───────────────┘
                               ▼
                ┌──────────────────────────────┐
                │  Wrap in Dataset + DataLoader│
                │      (batch_size = 32)       │
                └──────────────┬───────────────┘
                               ▼
                ┌──────────────────────────────┐
                │     Training Loop (20 ep.)   │
                │  forward → loss → backward   │
                │        → Adam update         │
                └──────────────┬───────────────┘
                               ▼
                ┌──────────────────────────────┐
                │   Evaluate on Test Set       │
                │       R² = 0.912             │
                └──────────────────────────────┘
```

### Training Log (Excerpt)

| Epoch | Avg RMSE |
|:---:|:---:|
| 1 | 36.520 |
| 5 | 8.476 |
| 10 | 7.138 |
| 15 | 6.192 |
| 20 | **5.700** |

The model converges rapidly, dropping from an RMSE of **36.5** to **5.7** in just 20 epochs.

---

## 📈 Results

| Metric | Value |
|---|---|
| **Test RMSE** (final epoch) | 5.700 |
| **Test R² Score** | **0.9124** |

An R² of **0.91** means the perceptron explains ~91 % of the variance in student performance using only four features — a strong result for a single-layer linear model.

---

## ⚙️ Requirements

| Package | Purpose |
|---|---|
| `Python 3.10+` | Runtime |
| `numpy` | Array operations |
| `pandas` | CSV loading & exploration |
| `torch` (PyTorch) | Tensor computation & autograd |
| `scikit-learn` | `r2_score` metric (evaluation only) |

Install everything with:

```bash
pip install numpy pandas torch scikit-learn
```

---

## ▶️ How to Run

1. **Clone the repository**

   ```bash
   git clone <repo-url>
   cd Scratch_projects
   ```

2. **Launch Jupyter**

   ```bash
   jupyter notebook Simple_perceptron.ipynb
   ```

3. **Run all cells** — the notebook trains the model and prints the R² score at the end.

---

## 🧩 Key Takeaways

- **No high-level abstractions** — weights, bias, forward pass, loss, and both optimisers are implemented from raw tensors and autograd, making every computation transparent.
- **Adam vs SGD** — both optimisers are available; Adam is used by default and shows significantly faster convergence.
- **Mini-batch training** — data is served through PyTorch's `DataLoader` in batches of 32, demonstrating real-world training patterns.
- **Solid baseline** — achieving R² ≈ 0.91 with a single linear layer validates the dataset's linear separability and provides a benchmark for more complex models.

---

## 📝 License

This project is for **educational and personal learning** purposes.
