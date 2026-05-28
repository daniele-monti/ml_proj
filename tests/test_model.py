import sys
from pathlib import Path


# Get the parent directory
parent_dir = Path(__file__).resolve().parent.parent

# Add parent directory to sys.path
sys.path.append(str(parent_dir))

from model import Model
from kernel_models import SVM, LogReg, Linear, Gaussian, Polynomial

def test_set_params_no_arguments():
    SUT = SVM()
    assert(hasattr(SUT, 'hello') == False)
    assert(hasattr(SUT, 'world') == False)
    SUT.set_params()
    assert(hasattr(SUT, 'hello') == False)
    assert(hasattr(SUT, 'world') == False)

def test_set_params_useless_arguments():
    SUT = SVM()
    assert(hasattr(SUT, 'hello') == False)
    assert(hasattr(SUT, 'world') == False)
    SUT.set_params(hello='hello', world='world')
    assert(hasattr(SUT, 'hello') == False)
    assert(hasattr(SUT, 'world') == False)

def test_set_params_useful_non_kernel_arguments():
    SUT = SVM()
    assert(hasattr(SUT, 'hello') == False)
    assert(hasattr(SUT, 'world') == False)
    SUT.set_params(tol=0.1, lambda_=0.0000001)
    assert(getattr(SUT, 'tol') == 0.1)
    assert(getattr(SUT, 'lambda_') == 0.0000001)

def test_set_params_useless_kernel_arguments_no_kernel_change():
    SUT = SVM(kernel='poly')
    assert(isinstance(getattr(SUT, '_kernel'), Polynomial))
    SUT.set_params(kernel='poly', hello=0.0000001)
    assert(isinstance(getattr(SUT, '_kernel'), Polynomial))
    assert(hasattr(SUT, 'hello') == False)
    assert(hasattr( getattr(SUT, '_kernel'), 'hello' ) == False)

def test_set_params_useless_kernel_arguments_with_kernel_change():
    SUT = SVM(kernel='poly')
    assert(isinstance(getattr(SUT, '_kernel'), Polynomial))
    SUT.set_params(kernel='linear', hello=0.0000001)
    assert(isinstance(getattr(SUT, '_kernel'), Linear))
    assert(hasattr(SUT, 'hello') == False)
    assert(hasattr( getattr(SUT, '_kernel'), 'hello' ) == False)

def test_set_params_useful_kernel_arguments_on_wrong_kernel_no_kernel_change():
    SUT = SVM(kernel='linear')
    assert(isinstance(getattr(SUT, '_kernel'), Linear))
    SUT.set_params(kernel='linear', gamma=0.0000001)
    assert(isinstance(getattr(SUT, '_kernel'), Linear))
    assert(hasattr(SUT, 'gamma') == False)
    assert(hasattr( getattr(SUT, '_kernel'), 'gamma' ) == False)

def test_set_params_useful_kernel_arguments_on_wrong_kernel_with_kernel_change():
    SUT = SVM(kernel='linear')
    assert(isinstance(getattr(SUT, '_kernel'), Linear))
    SUT.set_params(kernel='poly', gamma=0.73)
    assert(isinstance(getattr(SUT, '_kernel'), Polynomial))
    assert(hasattr(SUT, 'degree') == False)
    assert(hasattr(SUT, 'gamma') == False)
    assert(hasattr( getattr(SUT, '_kernel'), 'gamma' ) == False)
    assert(getattr( getattr(SUT, '_kernel'), 'degree' ) == 2)

def test_set_params_useful_kernel_arguments_on_right_kernel():
    SUT = SVM(kernel='linear')
    assert(isinstance(getattr(SUT, '_kernel'), Linear))
    SUT.set_params(kernel='rbf', gamma=0.0000001)
    assert(isinstance(getattr(SUT, '_kernel'), Gaussian))
    assert(getattr( getattr(SUT, '_kernel'), 'gamma' ) == 0.0000001)
