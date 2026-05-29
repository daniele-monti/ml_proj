import sys
from pathlib import Path

# Get the parent directory
parent_dir = Path(__file__).resolve().parent.parent

# Add parent directory to sys.path
sys.path.append(str(parent_dir))

from preprocessing import IQRClipper
import numpy as np


def test_clipping_on_same_set():
    example = np.array([
        [ 1,     1,     1,     1,     1,     1,     10,    100,   1000 ],
        [ 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000],
        [-10,   -2,     1,     1,     1,     2,     2,     2,     10   ],
        [ 1,     1,     20,    40,    50,    51,    100,   100,   100  ],
    ]).T
    SUT = IQRClipper()
    expected = np.array([
        [ 1,     1,     1,     1,     1,     1,     10,    23.5,  23.5 ],
        [ 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000],
        [-0.5,  -0.5,   1,     1,     1,     2,     2,     2,     3.5  ],
        [ 1,     1,     20,    40,    50,    51,    100,   100,   100  ],
    ]).T

    actual = SUT.fit_transform(example)
    np.testing.assert_allclose(actual, expected, atol=1e-4)


def test_clipping_on_different_set():
    example = np.array([
        [ 1,     1,     1,     1,     1,     1,     10,    100,   1000 ],
        [ 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000],
        [-10,   -2,     1,     1,     1,     2,     2,     2,     10   ],
        [ 1,     1,     20,    40,    50,    51,    100,   100,   100  ],
    ]).T
    SUT = IQRClipper()
    SUT.fit(example)

    other_set = np.array([
        [ 1000,  1000,  1000,  1000,  1000,  1000,  1000,  1000,  1000, 1000, 1000], 
        [ 1,     2,     3,     4,     5,     6,     7,     8,     9,    10,   11  ], 
        [-20,   -10,    0,     1,     1,     5,     10,    20,    1010, 45,  -7   ],
        [ 1,     1,     20,    40,    50,    51,    100,   100,   100,  200,  0   ],
    ]).T

    expected = np.array([
        [ 23.5,   23.5,  23.5,  23.5,  23.5,  23.5,  23.5,  23.5,  23.5,  23.5,  23.5 ],
        [ 10000,  10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000],
        [-0.5,   -0.5,   0,     1,     1,     3.5,   3.5,   3.5,   3.5,   3.5,  -0.5  ],
        [ 1,      1,     20,    40,    50,    51,    100,   100,   100,   200,   0    ],
    ]).T

    actual = SUT.transform(other_set)
    np.testing.assert_allclose(actual, expected, atol=1e-4)
