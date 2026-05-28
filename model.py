from metrics import ScoreMetrics


class Model:
    def __init__(self, **params):
        return
    
    def set_params(self, **params):
        for name, param in params.items():
            if hasattr(self, name):
                setattr(self, name, param)
    
    def fit(self, X, Y):
        return
    
    def predict(self, X_test):
        return
    
    def score(self, X_test, Y_test):
        prediction = self.predict(X_test)
        truth = Y_test
        return ScoreMetrics(truth, prediction)
