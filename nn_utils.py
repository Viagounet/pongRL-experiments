import numpy as np
from typing import TypedDict, Literal, Optional

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
    activations = []
    if values.ndim == 1:
        values = values.reshape(1, -1)
    for i, (weights, bias) in enumerate(layers):
        values = values @ weights + bias
        if i < len(layers) - 1:
            values = np.maximum(0, values)
        activations.append(values)
    return values, activations

def relu_derivative(preactivations):
    return np.where(preactivations > 0, 1, 0)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def mse(y_true, y_pred):
    return sum((y_true - y_pred)**2) / y_true.shape[0]

def update_params_randomly(params):
    new_params = []
    for weights, bias in params:
        weights = weights + rng.normal(0, 0.1, weights.shape)
        bias = bias + rng.normal(0, 0.1, bias.shape)
        new_params.append((weights, bias))
    return new_params


class HyperParameters(TypedDict):
    initial_lr: float
    lr_regime: Literal["constant", "linear"]
    final_lr: Optional[float]

def get_lr(current_step, max_step, hyperparams) -> float:
    match hyperparams["lr_regime"]:
        case "constant":
            return hyperparams["initial_lr"]
        case "linear":
            if hyperparams["final_lr"] is None:
                raise ValueError("With a `linear` lr regime, you need to set a final learning rate.")
            return np.interp(current_step, [0, max_step], [hyperparams["initial_lr"], hyperparams["final_lr"]])
        case _:
            raise ValueError("Parameter `lr_regime` must be either `constant` or `linear`")

def update_params_with_grads(params, grads, current_step:int, max_step: int, hyperparams: HyperParameters):
    new_params = []
    lr = get_lr(current_step, max_step, hyperparams)
    for (weights, bias), (grad_w, grad_b) in zip(params, grads):
        weights = weights - (grad_w * lr)
        bias = bias - (grad_b*lr)
        new_params.append((weights, bias))
    return new_params

def return_partial_derivatives(params, activations, x_train, y_train, y_pred):
    grads = []
    for i in range(len(params) - 1, -1, -1):
        weights, bias = params[i]
        input_to_layer = activations[i-1] if i > 0 else x_train
        preactivations = input_to_layer @ weights + bias # (batch_size, layer_dim)
        if i+1 == len(params): # In the final activations
            delta = 2*((y_pred - y_train)) / y_train.shape[0]
        else: # In the hidden layer
            delta = (delta @ params[i+1][0].T) * relu_derivative(preactivations)
        grad_w = input_to_layer.T @ delta
        grad_b = delta.sum(axis=0, keepdims=True)
        grads.insert(0, (grad_w, grad_b))
    return grads


if __name__ == "__main__":
    params = initialize_linear_nn_params((1,4,4,4,4,1))

    n_train= 1000
    n_test = 100

    noise_level = 0.1
    x_min, x_max = -np.pi, np.pi
    x_train = rng.uniform(x_min, x_max, (1000, 1))
    y_train = np.sin(x_train) + rng.normal(0, noise_level, x_train.shape)

    x_test = rng.uniform(x_min, x_max, (100, 1))
    y_test = np.sin(x_test) + rng.normal(0, noise_level, x_test.shape)


    MAX_STEP = 5000
    HYPERPARAMS = {"initial_lr": 0.05, "final_lr": 0, "lr_regime": "linear"}
    training_loss_history = []
    test_loss_history = []
    for step in range(MAX_STEP):
        pred, activations = infer(x_train, params)
        train_loss = mse(y_train, pred).sum()
        training_loss_history.append(train_loss)

        test_pred, test_activations = infer(x_test, params)
        test_loss = mse(y_test, test_pred).sum()
        test_loss_history.append(test_loss)
        
        grads = return_partial_derivatives(params, activations, x_train, y_train, pred)
        params = update_params_with_grads(params, grads, step, MAX_STEP, HYPERPARAMS)

    y_test_pred, activations = infer(x_test, params)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(range(MAX_STEP), training_loss_history, label='Train Loss', color='blue', linewidth=2)
    plt.plot(range(MAX_STEP), test_loss_history, label='Test Loss', color='orange', linewidth=2)
    plt.title("Loss over time")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.savefig("exports/nn_utils_test/loss.png")

    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, s=1, c='blue', label='Train (1000)')
    plt.scatter(x_test, y_test, s=15, c='red', marker='x', label='Test (100)')
    plt.scatter(x_test, y_test_pred, s=15, c='green', marker='x', label='Prediction (100)')
    plt.legend()
    plt.title("Données pour y = sin(x)")
    plt.savefig(f"exports/nn_utils_test/sinx_test.png")