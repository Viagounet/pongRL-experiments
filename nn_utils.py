import numpy as np

rng = np.random.default_rng(seed=1)

def initialize_linear_nn_params(shapes: tuple[int, ..., int]):
    layers = []
    previous_dim = shapes[0]
    for next_dim in shapes[1:]:
        weights = rng.random((previous_dim, next_dim)) - 0.5
        bias = rng.random((1, next_dim)) - 0.5
        layers.append((weights, bias))
        previous_dim = next_dim
    return layers

def infer(values: np.array, layers: list[tuple[np.array, np.array]]) -> np.array:
    if values.ndim == 1:
        values = values.reshape(1, -1)
    for i, (weights, bias) in enumerate(layers):
        values = values @ weights + bias
        if i < len(layers) - 1:
            values = np.maximum(0, values)
    return values


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

ACTION_SPACE = 1
params = initialize_linear_nn_params((1,10,10,10,10,1))

noise_level = 0.1
x_min, x_max = -np.pi, np.pi

def mse(y_true, y_pred):
    return sum((y_true - y_pred)**2) / y_true.shape[0]

def update_params_randomly(params):
    new_params = []
    for weights, bias in params:
        weights = weights + rng.normal(0, 0.1, weights.shape)
        bias = bias + rng.normal(0, 0.1, bias.shape)
        new_params.append((weights, bias))
    return new_params

x_train = rng.uniform(x_min, x_max, (1000, 1))
y_train = np.sin(x_train) + rng.normal(0, noise_level, x_train.shape)

x_test = rng.uniform(x_min*2, x_max*2, (100, 1))
y_test = np.sin(x_test) + rng.normal(0, noise_level, x_test.shape)

best_params = params
y_pred = infer(x_train, best_params)
best_loss = mse(y_train, y_pred)
for i in range(100):
    new_params = update_params_randomly(best_params)
    y_pred = infer(x_train, new_params)
    loss = mse(y_train, y_pred)
    if loss < best_loss:
        best_loss = loss
        best_params = new_params

y_test_pred = infer(x_test, best_params)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, s=1, c='blue', label='Train (1000)')
plt.scatter(x_test, y_test, s=15, c='red', marker='x', label='Test (100)')
plt.scatter(x_test, y_test_pred, s=15, c='green', marker='x', label='Prediction (100)')
plt.legend()
plt.title("Données pour y = sin(x)")
plt.savefig(f"preds.png")