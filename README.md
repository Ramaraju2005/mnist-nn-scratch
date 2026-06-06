# Neural Network from Scratch — MNIST Digit Classifier

Built a fully connected neural network **using only NumPy** (no PyTorch, no TensorFlow) to classify handwritten digits from the MNIST dataset.

This was my attempt to actually understand what happens inside a neural network — forward pass, backpropagation, gradient descent — by implementing every piece of math manually instead of calling `.fit()`.

---

## Why I built this

I had been using ML libraries for a while (scikit-learn, HuggingFace) without really understanding what was happening under the hood. I wanted to answer questions like:

- How does a network actually "learn"?
- What does backpropagation compute, step by step?
- Why does weight initialization matter?

Building this from scratch forced me to understand the math, not just the API.

---

## Model Architecture

```
Input Layer   →   784 neurons   (28×28 pixel image, flattened)
Hidden Layer  →   128 neurons   (ReLU activation)
Output Layer  →   10 neurons    (Softmax — one per digit class)
```

No external deep learning library is used. Every operation — matrix multiplication, activation, loss, gradient — is written manually in NumPy.

---

## What I implemented manually

**Forward Pass**
- Linear transformation: `Z = W · X + b`
- ReLU activation: `max(0, Z)`
- Softmax output for probabilities

**Loss Function**
- Cross-Entropy Loss: measures how wrong the predictions are

**Backpropagation**
- Chain rule applied layer by layer
- Gradients computed for weights and biases at each layer

**Weight Update**
- Stochastic Gradient Descent (SGD): `W = W - lr * dW`

**Weight Initialization**
- He initialization: `W ~ N(0, sqrt(2/n))` — works well with ReLU

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | ~96% |
| Epochs | 20 |
| Batch Size | 64 |
| Learning Rate | 0.01 |
| Activation | ReLU |

---

## How to run

```bash
# Install dependencies
pip install -r requirements.txt

# Train with default settings
python train.py

# Try different settings
python train.py --activation sigmoid --epochs 30 --lr 0.005 --hidden_size 256
```

**Arguments:**

| Argument | Default | Options |
|---|---|---|
| `--activation` | `relu` | `relu`, `sigmoid` |
| `--epochs` | `20` | any integer |
| `--lr` | `0.01` | any float |
| `--batch_size` | `64` | any integer |
| `--hidden_size` | `128` | any integer |

---

## Project Structure

```
mnist-nn-scratch/
├── model.py          ← Neural network class (forward, backward, activations)
├── train.py          ← Training loop, data loading, evaluation
├── requirements.txt  ← Dependencies
├── results/
│   ├── loss_history.npy
│   ├── accuracy_history.npy
│   └── summary.txt
└── README.md
```

---

## Key things I learned

1. **Backpropagation is just the chain rule** applied repeatedly from output to input. Once I wrote out the math for each layer, it made sense.

2. **Softmax + Cross-Entropy gradient simplifies nicely** — the derivative at the output layer is just `(predicted - actual)`, which is clean and intuitive.

3. **Weight initialization matters a lot.** Using random small weights (He init) made training stable. Initializing to zeros causes all neurons to learn the same thing (symmetry problem).

4. **Vectorization is essential.** Computing gradients across the full batch using matrix operations instead of loops is what makes it fast.

5. **ReLU outperforms Sigmoid** on this task because sigmoid gradients get very small in deep layers (vanishing gradient), while ReLU passes gradients cleanly.

---

## What's next

- [ ] Add a second hidden layer to see if accuracy improves
- [ ] Implement Momentum / Adam optimizer
- [ ] Add dropout regularization
- [ ] Try on Fashion-MNIST

---

## Dependencies

- `numpy` — all math operations
- `scikit-learn` — only used to download the MNIST dataset
- `matplotlib` — plotting loss/accuracy curves

---

## References

- [Andrew Ng — Neural Networks and Deep Learning (Coursera)](https://www.coursera.org/learn/neural-networks-deep-learning)
- [CS231n Stanford — Backpropagation Notes](https://cs231n.github.io/optimization-2/)
- [3Blue1Brown — But what is a Neural Network?](https://www.youtube.com/watch?v=aircAruvnKk)
