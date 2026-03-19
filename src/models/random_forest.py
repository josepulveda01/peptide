# random_forest.py
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class RandomForestWithUncertainty:
    """
    Random Forest Regressor con estimación de incertidumbre.
    
    Por defecto, la incertidumbre corresponde a la desviación estándar
    del promedio del ensemble (std / sqrt(n_estimators)), lo cual refleja
    la confianza en la predicción final.
    """
    
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators = n_estimators,
            max_depth = max_depth,
            random_state = random_state,
            n_jobs = -1  # utiliza todos los cores de la CPU
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X, return_tree_std=False):
        predicts = np.array([tree.predict(X) for tree in self.model.estimators_])
        mean = predicts.mean(axis=0)

        # Desviación estándar del promedio (incertidumbre del ensemble)
        std = predicts.std(axis=0) / np.sqrt(len(self.model.estimators_))

        if return_tree_std:
            tree_std = predicts.std(axis=0)  # std muestral de los árboles
            return mean, std, tree_std

        return mean, std
