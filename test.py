import numpy as np


color = np.arange(5*5*3).reshape(5, 5, 3).astype(np.uint8)
depth = np.arange(5*5).reshape(5, 5).astype(np.float32)

np.savez("test.npz", color=color, depth=depth)
