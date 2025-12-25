import numpy as np

def frames_to_bw_resized(frames, last_n, new_size: tuple[int, int]):
    bw_array = np.mean(frames[-last_n:], axis=(0, 3))
    h_idx = np.linspace(0, bw_array.shape[0] - 1, new_size[0]).astype(int)
    w_idx = np.linspace(0, bw_array.shape[1] - 1, new_size[1]).astype(int)
    resized = bw_array[h_idx][:, w_idx]
    return resized