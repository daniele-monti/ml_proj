import sys
from pathlib import Path
import numpy as np

# Get the parent directory
parent_dir = Path(__file__).resolve().parent.parent

# Add parent directory to sys.path
sys.path.append(str(parent_dir))

from metrics import ScoreMetrics


def test_confusion_matrix():
    truth =      np.array([-1, -1, 1,  1,  1, 1,  1,  1, -1, 1, -1, -1,  1, -1, -1, 1, 1, -1, -1, 1, 1])
    prediction = np.array([-1, -1, 1, -1, -1, 1, -1, -1, -1, 1,  1,  1, -1, -1,  1, 1, 1, -1, -1, 1, 1])
    SUT = ScoreMetrics(truth, prediction)
    assert(SUT.true_neg == 6)
    assert(SUT.true_pos == 7)
    assert(SUT.false_neg == 5)
    assert(SUT.false_pos == 3)

def test_metrics():
    truth =      np.array([-1, -1, 1,  1,  1, 1,  1,  1, -1, 1, -1, -1,  1, -1, -1, 1, 1, -1, -1, 1, 1])
    prediction = np.array([-1, -1, 1, -1, -1, 1, -1, -1, -1, 1,  1,  1, -1, -1,  1, 1, 1, -1, -1, 1, 1])
    SUT = ScoreMetrics(truth, prediction)
    np.testing.assert_allclose(SUT.accuracy, 0.6190476190, atol=1e-7)
    np.testing.assert_allclose(SUT.precision_pos, 0.7, atol=1e-7)
    np.testing.assert_allclose(SUT.recall_pos , 0.5833333333, atol=1e-7)
    np.testing.assert_allclose(SUT.f1_score_pos, 0.6363636363, atol=1e-7)
    np.testing.assert_allclose(SUT.precision_neg, 0.5454545454, atol=1e-7)
    np.testing.assert_allclose(SUT.recall_neg, 0.666666666, atol=1e-7)
    np.testing.assert_allclose(SUT.f1_score_neg, 0.6, atol=1e-7)

def test_aggregate():
    list = []

    truth =      np.array([-1, -1, 1,  1,  1, 1,  1,  1,])
    prediction = np.array([-1, -1, 1, -1, -1, 1, -1, -1,])
    list.append(ScoreMetrics(truth, prediction))

    truth =      np.array([-1, 1, -1, -1,  1, -1, -1, 1, 1, -1, -1, 1, 1])
    prediction = np.array([-1, 1,  1,  1, -1, -1,  1, 1, 1, -1, -1, 1, 1])
    list.append(ScoreMetrics(truth, prediction))
    
    SUT = ScoreMetrics.aggregate(list)
    np.testing.assert_allclose(SUT.true_neg, 3.0, atol=1e-7)
    np.testing.assert_allclose(SUT.true_pos, 3.5, atol=1e-7)
    np.testing.assert_allclose(SUT.false_neg , 2.5, atol=1e-7)
    np.testing.assert_allclose(SUT.false_pos, 1.5, atol=1e-7)
    np.testing.assert_allclose(SUT.accuracy, 0.5961538461, atol=1e-7)
    np.testing.assert_allclose(SUT.precision_pos, 0.8125, atol=1e-7)
    np.testing.assert_allclose(SUT.recall_pos , 0.5833333333, atol=1e-7)
    np.testing.assert_allclose(SUT.f1_score_pos, 0.6071428571, atol=1e-7)
    np.testing.assert_allclose(SUT.precision_neg, 0.5666666666, atol=1e-7)
    np.testing.assert_allclose(SUT.recall_neg, 0.7857142857, atol=1e-7)
    np.testing.assert_allclose(SUT.f1_score_neg, 0.583333333, atol=1e-7)
