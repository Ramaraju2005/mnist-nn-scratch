import numpy as np
import argparse
import os
import time
from model import NeuralNetwork

# ─────────────────────────────────────────
# Data Loading & Preprocessing
# ─────────────────────────────────────────

def load_mnist():
    """Load MNIST via sklearn (avoids manual download)."""
    from sklearn.datasets import fetch_openml
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target.astype(int)
    return X, y


def preprocess(X, y):
    """Normalize pixels to [0,1] and one-hot encode labels."""
    X = X / 255.0

    # One-hot encode
    y_onehot = np.zeros((y.shape[0], 10))
    y_onehot[np.arange(y.shape[0]), y] = 1

    return X, y_onehot


def train_val_test_split(X, y_onehot):
    """60k train / 10k test split (standard MNIST)."""
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y_onehot[:60000], y_onehot[60000:]

    # Further split 5k from train as validation
    X_val, y_val = X_train[:5000], y_train[:5000]
    X_train, y_train = X_train[5000:], y_train[5000:]

    return X_train, y_train, X_val, y_val, X_test, y_test


# ─────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────

def train(model, X_train, y_train, X_val, y_val,
          epochs=20, batch_size=64, learning_rate=0.01):

    num_samples = X_train.shape[0]
    num_batches = num_samples // batch_size

    print(f"\nTraining for {epochs} epochs | batch_size={batch_size} | lr={learning_rate}")
    print(f"Training samples: {num_samples} | Validation samples: {X_val.shape[0]}\n")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>10} | {'Val Acc':>9} | {'Time':>7}")
    print("-" * 56)

    for epoch in range(1, epochs + 1):
        start = time.time()

        # Shuffle training data
        perm = np.random.permutation(num_samples)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]

        epoch_loss = 0

        for j in range(num_batches):
            start_idx = j * batch_size
            end_idx = min(start_idx + batch_size, num_samples)

            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]

            # Forward
            output = model.forward(X_batch)

            # Loss
            loss = model.cross_entropy_loss(output, y_batch)
            epoch_loss += loss

            # Backward
            model.backward(y_batch)

            # Update
            model.update_params(learning_rate)

        # Metrics
        avg_loss = epoch_loss / num_batches
        train_output = model.forward(X_train[:5000])
        train_acc = model.accuracy(train_output, y_train[:5000])

        val_output = model.forward(X_val)
        val_acc = model.accuracy(val_output, y_val)

        elapsed = time.time() - start

        model.loss_history.append(avg_loss)
        model.accuracy_history.append(val_acc)

        print(f"{epoch:>6} | {avg_loss:>10.4f} | {train_acc*100:>9.2f}% | {val_acc*100:>8.2f}% | {elapsed:>5.1f}s")

    print("-" * 56)


# ─────────────────────────────────────────
# Save Results
# ─────────────────────────────────────────

def save_results(model, test_acc):
    os.makedirs("results", exist_ok=True)

    # Save loss + accuracy history
    np.save("results/loss_history.npy", model.loss_history)
    np.save("results/accuracy_history.npy", model.accuracy_history)

    # Save final test accuracy
    with open("results/summary.txt", "w") as f:
        f.write(f"Final Test Accuracy: {test_acc*100:.2f}%\n")
        f.write(f"Total Epochs: {len(model.loss_history)}\n")
        f.write(f"Final Val Accuracy: {model.accuracy_history[-1]*100:.2f}%\n")

    print(f"\nResults saved to results/")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Neural Network from Scratch on MNIST")
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'sigmoid'],
                        help='Activation function for hidden layer')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=128, help='Number of hidden neurons')
    args = parser.parse_args()

    # Load & preprocess
    X, y = load_mnist()
    X, y_onehot = preprocess(X, y)
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y_onehot)

    # Build model
    model = NeuralNetwork(
        input_size=784,
        hidden_size=args.hidden_size,
        output_size=10,
        activation=args.activation
    )

    print(f"\nModel Architecture:")
    print(f"  Input Layer  : 784 neurons (28x28 flattened)")
    print(f"  Hidden Layer : {args.hidden_size} neurons ({args.activation})")
    print(f"  Output Layer : 10 neurons (softmax)")

    # Train
    train(model, X_train, y_train, X_val, y_val,
          epochs=args.epochs,
          batch_size=args.batch_size,
          learning_rate=args.lr)

    # Final test evaluation
    test_output = model.forward(X_test)
    test_acc = model.accuracy(test_output, y_test)
    print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

    save_results(model, test_acc)


if __name__ == "__main__":
    main()
