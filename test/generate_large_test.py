import os

import numpy as np

if __name__ == "__main__":
    test_int = np.arange(1000000).astype(np.int32)
    test_int = test_int.reshape(200, 5, 1000)
    test_float = np.arange(1000000).astype(np.float32)
    test_float = test_float.reshape(1000, 5, 20, 10)

    path = os.path.join(os.path.dirname(__file__), "..",
                        "assets", "test", "test_large.npz")
    np.savez(path, test_int=test_int, test_float=test_float)

    path = os.path.join(os.path.dirname(__file__), "..",
                        "assets", "test", "test_large_compressed.npz")
    np.savez_compressed(path, test_int=test_int, test_float=test_float)
