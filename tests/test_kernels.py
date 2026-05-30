import sys
from pathlib import Path

# Get the parent directory
parent_dir = Path(__file__).resolve().parent.parent

# Add parent directory to sys.path
sys.path.append(str(parent_dir))

import kernel_models as models
import numpy as np

def test_linear():
    SUT = models.KernelFactory.get_kernel("linear")
    mat = np.array([
        [1. , 2. ,  0. , 2.2],
        [3. , 8. , 10. , 1. ],
        [1.5, 1. ,  0.3, 5. ],
    ])
    expected_gram = np.array([
        [10.84, 22.2, 15.5],
        [22.2, 175.0, 21.5],
        [15.5, 21.5, 29.34]
    ])
    np.testing.assert_allclose(SUT.gram(mat, mat), expected_gram, atol=1e-7)

def test_gaussian_1():
    SUT = models.KernelFactory.get_kernel("rbf", gamma=1)
    mat = np.array([
        [1. , 2. ,  0. , 2.2],
        [3. , 8. , 10. , 1. ],
        [1.5, 1. ,  0.3, 5. ],
    ])
    expected_gram = np.array([
        [2.0, 1.0, 1.00010308053314970452],
        [1.0, 2.0, 1.0],
        [1.00010308053314970452, 1.0, 2.0]
    ])
    np.testing.assert_allclose(SUT.gram(mat, mat), expected_gram, atol=1e-7)

def test_gaussian_01():
    SUT = models.KernelFactory.get_kernel("rbf", gamma=0.1)
    mat = np.array([
        [1. , 2. ,  0. , 2.2],
        [3. , 8. , 10. , 1. ],
        [1.5, 1. ,  0.3, 5. ],
    ])
    expected_gram = np.array([
        [2.0, 1.00000072001053, 1.3993168767363899],
        [1.00000072001053, 2.0, 1.0000000984221458],
        [1.3993168767363899, 1.0000000984221458, 2.0]
    ])
    np.testing.assert_allclose(SUT.gram(mat, mat), expected_gram, atol=1e-7)

def test_poly_3():
    SUT = models.KernelFactory.get_kernel("poly", degree=3)
    mat = np.array([
        [1. , 2. ,  0. , 2.2],
        [3. , 8. , 10. , 1. ],
        [1.5, 1. ,  0.3, 5. ],
    ])
    expected_gram = np.array([
        [1274.760704, 10942.047999999999, 3724.875],
        [10942.047999999999, 5359376.0, 9939.375],
        [3724.875, 9939.375, 25257.916504]
    ])
    np.testing.assert_allclose(SUT.gram(mat, mat), expected_gram, atol=1e-7)

def test_poly_10():
    SUT = models.KernelFactory.get_kernel("poly", degree=10)
    mat = np.array([
        [1. , 2. ,  0. , 2.2],
        [3. , 8. , 10. , 1. ],
        [1.5, 1. ,  0.3, 5. ],
    ])
    expected_gram = np.array([
        [22402311000.694443, 29075670897347.71, 800418249005.6885],
        [29075670897347.71, 2.6938938999176025e+22, 21104963196567.65],
        [800418249005.6885, 21104963196567.65, 472716863126707.9]
    ])
    np.testing.assert_allclose(SUT.gram(mat, mat), expected_gram, atol=1e-7)
