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
    values = values.reshape(1,-1)
    for i, (weights, bias) in enumerate(layers):
        values = values @ weights + bias
        if i < len(layers) - 1:
            values = np.maximum(0, values)
    return values


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

ACTION_SPACE = 6
params = initialize_linear_nn_params((2,3,ACTION_SPACE))
init_vector = np.array([2,3])
activations = infer(init_vector, params)
actions_probs = softmax(activations)
print(actions_probs)