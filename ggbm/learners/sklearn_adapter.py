# learners/sklearn_adapter.py

class SklearnRegressorAdapter:
    def __init__(self, estimator):
        self.estimator = estimator
    def fit(self, X, y, sample_weight=None):
        kw = {}
        if hasattr(self.estimator, "fit"):
            if "sample_weight" in self.estimator.fit.__code__.co_varnames:
                kw["sample_weight"] = sample_weight
        self.estimator.fit(X, y, **kw)

        return self

    def predict(self, X):
        return self.estimator.predict(X)