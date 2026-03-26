<div align="center">

# 🔢 Deep Learning — Part 2: Convolutional Neural Networks (CNN)

### MNIST Handwritten Digit Classification using CNN

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

<br/>

> **A complete end-to-end deep learning pipeline for MNIST handwritten digit classification — from data loading and preprocessing, through model training and evaluation, to a real-world predictive system that classifies custom digit images using OpenCV.**

</div>

---

## 📋 Table of Contents
- [About the Project](#-about-the-project)
- [Demo](#-demo)
- [Tech Stack](#-tech-stack)
- [Features](#-features)
- [Dataset](#-dataset)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Contributing](#-contributing)
- [License](#-license)

---

## 📌 About the Project

**Deep Learning — Part 3** focuses on **image classification using Neural Networks** applied to the iconic **MNIST dataset** — 70,000 grayscale images of handwritten digits (0–9).

The notebook covers the complete ML workflow in a single, well-structured file:

| Section | What It Covers |
|---------|----------------|
| 📥 **Data Loading** | Load MNIST directly from `keras.datasets` |
| 🔍 **EDA** | Explore shapes, unique labels, and visualize sample digits |
| ⚙️ **Preprocessing** | Pixel normalization (0–255 → 0–1) |
| 🧠 **Model Building** | Flatten → Dense(50) → Dense(50) → Dense(10) |
| 📈 **Training & Evaluation** | 10 epochs, Adam optimizer, ~98.9% train / ~97.1% test accuracy |
| 🔥 **Confusion Matrix** | Seaborn heatmap of all 10 digit classes |
| 🖼️ **Predictive System** | Load a real PNG digit image → preprocess with OpenCV → predict |

> 💡 The project also includes a **real-world predictive system** that takes any external digit image (`MNIST_digit.png`), preprocesses it with OpenCV (grayscale conversion + resize to 28×28 + normalization), and returns the predicted digit label.

---

## 🌐 Demo

### 🧠 Model Performance

| Metric | Value |
|--------|-------|
| **Training Accuracy** | ~98.9% |
| **Testing Accuracy** | ~97.1% |
| **Epochs** | 10 |
| **Optimizer** | Adam |
| **Loss Function** | Sparse Categorical Cross-Entropy |

---

### 🖼️ Predictive System — Custom Image Input

The notebook includes a full predictive pipeline for real-world digit images:

```
Input: MNIST_digit.png  (any handwritten digit image)
         │
         ▼
   OpenCV imread → BGR to RGB display
         │
         ▼
   Grayscale conversion → Resize to (28, 28)
         │
         ▼
   Normalize (÷ 255) → Reshape to [1, 28, 28]
         │
         ▼
   model.predict() → np.argmax() → Predicted Digit Label
```

**Example:** The included `MNIST_digit.png` (digit "3") is correctly classified by the trained model.

---

### 🔥 Confusion Matrix

A 10×10 seaborn heatmap visualizes model performance across all digit classes, making it easy to spot which digits are occasionally confused (e.g., 4 vs 9, 3 vs 8).

---

## 🛠️ Tech Stack

| Technology | Role |
|------------|------|
| **Python 3.10** | Core language |
| **TensorFlow / Keras** | Model building, training, evaluation, MNIST data loading |
| **NumPy** | Array operations, argmax predictions, reshaping |
| **OpenCV (cv2)** | Image loading, BGR→RGB, grayscale conversion, resizing |
| **Matplotlib** | Image visualization, training output display |
| **Seaborn** | Confusion matrix heatmap |
| **PIL (Pillow)** | Supplementary image handling |

---

## ✨ Features

<details open>
<summary><b>📥 Data Loading & Exploration</b></summary>
<br/>

- MNIST loaded directly from `keras.datasets` — no manual download needed
- Train/Test split: **60,000** training images + **10,000** test images
- Shape inspection: `(60000, 28, 28)` for images, `(60000,)` for labels
- Sample digit visualization with `plt.imshow()`
- Unique label verification: digits `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`

</details>

<details open>
<summary><b>⚙️ Preprocessing</b></summary>
<br/>

- **Pixel normalization** — scale all pixel values from `[0, 255]` to `[0.0, 1.0]` by dividing by 255
- No manual resizing needed — all MNIST images are uniformly `28×28`
- Labels used as-is with `sparse_categorical_crossentropy` — no one-hot encoding required

</details>

<details open>
<summary><b>🧠 Neural Network Architecture</b></summary>
<br/>

- **Flatten layer** — converts 28×28 image to 784-dimensional vector
- **Dense(50, ReLU)** — first hidden layer
- **Dense(50, ReLU)** — second hidden layer
- **Dense(10, Sigmoid)** — output layer with 10 neurons (one per digit class)
- Compiled with **Adam** optimizer + **Sparse Categorical Cross-Entropy** loss

</details>

<details open>
<summary><b>📈 Training & Evaluation</b></summary>
<br/>

- Trained for **10 epochs** on 60,000 images
- Training accuracy: **~98.9%**
- Test accuracy: **~97.1%** via `model.evaluate()`
- `model.predict()` returns probability distribution across 10 classes
- `np.argmax()` used to convert probabilities to final class labels

</details>

<details open>
<summary><b>🔥 Confusion Matrix</b></summary>
<br/>

- Full **10×10 confusion matrix** computed using `tensorflow.math.confusion_matrix`
- Visualized as a **seaborn heatmap** (`figsize=(15,7)`, annotated, Blues colormap)
- Axes labelled: `True Labels` (Y) vs `Predicted Labels` (X)

</details>

<details open>
<summary><b>🖼️ Real-World Predictive System</b></summary>
<br/>

- Loads any external digit PNG using **OpenCV**
- Converts BGR → RGB for display, then → Grayscale for processing
- Resizes to `(28, 28)` to match MNIST input format
- Normalizes pixel values and reshapes to `[1, 28, 28]`
- Runs `model.predict()` → `np.argmax()` → prints the predicted digit
- Tested on included `MNIST_digit.png` (digit **"3"**)

</details>

---

## 📊 Dataset

### MNIST Dataset

| Property | Value |
|----------|-------|
| **Training Samples** | 60,000 |
| **Testing Samples** | 10,000 |
| **Image Size** | 28 × 28 pixels |
| **Color** | Grayscale |
| **Classes** | 10 (digits 0–9) |
| **Pixel Range** | 0 – 255 (normalized to 0.0 – 1.0) |
| **Source** | Built into `keras.datasets.mnist` |

```python
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# X_train: (60000, 28, 28) | Y_train: (60000,)
# X_test:  (10000, 28, 28) | Y_test:  (10000,)
```

### Custom Digit Image (`MNIST_digit.png`)

| Property | Value |
|----------|-------|
| **Content** | Handwritten digit "3" |
| **Format** | PNG |
| **Processing** | BGR → Grayscale → Resize (28×28) → Normalize → Reshape [1,28,28] |
| **Predicted Label** | `3` ✅ |

---

## 🧠 How It Works

### Full End-to-End Pipeline

```
  MNIST Dataset (keras.datasets)
        │
        ▼
  ┌──────────────────────────────────────┐
  │  Load Data                           │
  │  X_train (60000,28,28)               │
  │  X_test  (10000,28,28)               │
  └──────────────────────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────┐
  │  Normalize Pixels                    │
  │  X_train = X_train / 255             │
  │  X_test  = X_test  / 255             │
  └──────────────────────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────┐
  │  Keras Sequential Model              │
  │  Flatten(28×28 → 784)                │
  │  → Dense(50, ReLU)                   │
  │  → Dense(50, ReLU)                   │
  │  → Dense(10, Sigmoid)                │
  └──────────────────────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────┐
  │  Compile & Train                     │
  │  Optimizer : Adam                    │
  │  Loss      : Sparse Categorical CE   │
  │  Epochs    : 10                      │
  └──────────────────────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────┐
  │  Evaluate on Test Set                │
  │  Accuracy ≈ 97.1%                    │
  └──────────────────────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────┐
  │  Confusion Matrix                    │
  │  10×10 Seaborn Heatmap               │
  └──────────────────────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────┐
  │  Predictive System                   │
  │  Custom PNG → OpenCV → Predict       │
  │  Output: Digit Label (0–9)           │
  └──────────────────────────────────────┘
```

### Visual Output States

| Output | Description |
|--------|-------------|
| 🖼️ **Sample digit plot** | `plt.imshow(X_train[25])` — visualize any training image |
| 📊 **Confusion heatmap** | 10×10 matrix showing true vs predicted for all test samples |
| 🔢 **Predicted label** | `np.argmax(model.predict(image_reshaped))` → single digit (0–9) |

---

## 📁 Project Structure

```
Deep_Learning_Part_3/
│
├── 📓 MNIST_Handwritten_Digit_Clasification_using_CNN.ipynb   # Main notebook — full pipeline
├── 🖼️ MNIST_digit.png                                          # Sample digit image for prediction (digit "3")
│
└── 📖 README.md                                                # This file
```

### Key Files Explained

| File | Purpose |
|------|---------|
| `MNIST_Handwritten_Digit_Clasification_using_CNN.ipynb` | Complete notebook — data loading, EDA, preprocessing, model building, training, confusion matrix, and predictive system |
| `MNIST_digit.png` | External handwritten digit image (digit "3") used to test the real-world predictive system via OpenCV |

---

## ⚙️ Getting Started

### Prerequisites
- Python 3.10+
- Jupyter Notebook or JupyterLab
- pip

### 1 — Clone the Repository

```bash
git clone https://github.com/YourUsername/Deep_Learning_Part_3.git
cd Deep_Learning_Part_3
```

### 2 — Install Dependencies

```bash
pip install tensorflow opencv-python numpy matplotlib seaborn pillow jupyter
```

### 3 — Launch Jupyter

```bash
jupyter notebook
```

### 4 — Run the Notebook

```
MNIST_Handwritten_Digit_Clasification_using_CNN.ipynb
```

> ⚠️ Make sure `MNIST_digit.png` is in the **same directory** as the notebook before running the Predictive System section. Update the `input_image_path` variable to a relative path:
> ```python
> input_image_path = "MNIST_digit.png"
> ```

---

## 🤝 Contributing

Contributions are welcome! If you have improvements, new experiments, or want to extend this to a full CNN with Conv2D layers:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

<div align="center">

**Made with ❤️ for Deep Learning enthusiasts and beginners**

⭐ If this helped your learning journey, please give it a star!

</div>
