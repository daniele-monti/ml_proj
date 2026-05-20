import sys
from pathlib import Path

# Get the parent directory
parent_dir = Path(__file__).resolve().parent.parent

# Add parent directory to sys.path
sys.path.append(str(parent_dir))

import preprocessing as prep
import pandas as pd
import numpy as np

example = pd.DataFrame({
    '1': [ 1,     1,     1,     1,     1,     1,     10,    100,   1000 ],
    '2': [ 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000],
    '3': [-10,   -2,     1,     1,     1,     2,     2,     2,     10   ],
    '4': [ 1,     1,     20,    40,    50,    51,    100,   100,   100  ],
    '5': [ 1,     0,     0,     1,     0,     1,     0,     0,     0    ]
})

def test_clipping():
    SUT = prep.clip_outliers(example, columns=['1', '2', '3', '4'])
    expected = np.array([
        [ 1,     1,     1,     1,     1,     1,     10,    23.5,  23.5 ],
        [ 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000],
        [-0.5,  -0.5,   1,     1,     1,     2,     2,     2,     3.5  ],
        [ 1,     1,     20,    40,    50,    51,    100,   100,   100  ],
        [ 1,     0,     0,     1,     0,     1,     0,     0,     0    ]
    ])

    actual = SUT.to_numpy().T
    np.testing.assert_allclose(actual, expected)


def test_scaling_on_same_set():
    SUT = prep.Scaler(example, columns=['1', '3', '4', '5'])
    expected = np.array([
        [0,     0,     0,     0,     0,     0,     0.009, 0.099, 1    ],
        [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000],
        [0,     0.4,   0.55,  0.55,  0.55,  0.6,   0.6,   0.6,   1    ],
        [0,     0,     0.192, 0.394, 0.495, 0.505, 1,     1,     1    ],
        [1,     0,     0,     1,     0,     1,     0,     0,     0    ]
    ])

    actual = SUT.scale(example).to_numpy().T
    np.testing.assert_allclose(actual, expected, atol=1e-4)

def test_scaling_on_different_set():
    SUT = prep.Scaler(example, columns=['1', '3', '4', '5'])

    other_set = pd.DataFrame({
        '1': [ 1000,  1000,  1000,  1000,  1000,  1000,  1000,  1000,  1000 ], # all different but the same
        '2': [ 1,     2,     3,     4,     5,     6,     7,     8,     9    ], # this should be ignored
        '3': [-20,   -10,    0,     1,     1,     5,     10,    20,    1010 ], # this has an outlier
        '4': [ 1,     1,     20,    40,    50,    51,    100,   100,   100  ], # this should have the same scaling because it's the same row
        '5': [ 1,     0,     0,     1,     0,     1,     0,     0,     0    ]  # this should have the same scaling because it's the same row
    })

    expected = np.array([
        [ 1,     1,     1,     1,     1,     1,     1,     1,     1    ],
        [ 1,     2,     3,     4,     5,     6,     7,     8,     9    ],
        [-0.5,   0,     0.5,   0.55,  0.55,  0.75,  1,     1.5,   51   ],
        [ 0,     0,     0.192, 0.394, 0.495, 0.505, 1,     1,     1    ],
        [ 1,     0,     0,     1,     0,     1,     0,     0,     0    ]
    ])

    actual = SUT.scale(other_set).to_numpy().T
    np.testing.assert_allclose(actual, expected, atol=1e-4)