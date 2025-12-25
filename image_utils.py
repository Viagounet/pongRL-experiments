import numpy as np

from PIL import Image
from pathlib import Path

def save_rgb(array, path):
    img = Image.fromarray(array.astype(np.uint8), 'RGB')
    img.save(path)

def save_bw(array, path):
    img = Image.fromarray(array.astype(np.uint8), 'L')
    img.save(path)