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
        [9.84, 21.2, 14.5],
        [21.2, 174.0, 20.5],
        [14.5, 20.5, 28.34]
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
        [1.0, 3.7444538265689175e-62, 0.00010308053314970452],
        [3.7444538265689175e-62, 1.0, 8.529592626086446e-71],
        [0.00010308053314970452, 8.529592626086446e-71, 1.0]
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
        [1.0, 7.200105300311436e-07, 0.3993168767363899],
        [7.200105300311436e-07, 1.0, 9.842214572786463e-08],
        [0.3993168767363899, 9.842214572786463e-08, 1.0]
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
        [1273.760704, 10941.047999999999, 3723.875],
        [10941.047999999999, 5359375.0, 9938.375],
        [3723.875, 9938.375, 25256.916504]
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
        [22402310999.694443, 29075670897346.71, 800418249004.6885],
        [29075670897346.71, 2.6938938999176025e+22, 21104963196566.65],
        [800418249004.6885, 21104963196566.65, 472716863126706.9]
    ])
    np.testing.assert_allclose(SUT.gram(mat, mat), expected_gram, atol=1e-7)
