# Random forest

import numpy as np
from sklearn.ensemble import RandomForestRegressor


class RandomForestWithUncertainty:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1 # utiliza todos los cores de la CPU
        )

    def fit(self, X, y):
        self.model.fit(X, y) # entrenamiento

    def predict(self, X):
        predicts = np.array([tree.predict(X) for tree in self.model.estimators_])
        mean = predicts.mean(axis=0)
        std = predicts.std(axis=0)
        return mean, std


class MultiOutputModel:
    def __init__(self):
        self.affinity_model = RandomForestWithUncertainty()
        self.solubility_model = RandomForestWithUncertainty()

    def fit(self, X, y_aff, y_sol):
        self.affinity_model.fit(X, y_aff)
        self.solubility_model.fit(X, y_sol)

    def predict(self, X):
        mean_aff, std_aff = self.affinity_model.predict(X)
        mean_sol, std_sol = self.solubility_model.predict(X)

        return {
            "affinity_mean": mean_aff,
            "affinity_std": std_aff,
            "solubility_mean": mean_sol,
            "solubility_std": std_sol,
        }
