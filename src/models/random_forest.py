import numpy as np
from sklearn.ensemble import RandomForestRegressor

class RandomForestWithUncertainty:
    """
    Random Forest Regressor multi-output con estimación de incertidumbre.

    Predice múltiples targets (ej: afinidad y solubilidad) y estima
    la incertidumbre del ensemble.
    
    Returns:
    --------
    mean : np.ndarray de forma (n_peptidos, n_targets)
    std  : np.ndarray de forma (n_peptidos, n_targets)
    """

    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators = n_estimators,
            max_depth = max_depth,
            random_state = random_state,
            n_jobs = -1
        )

    def fit(self, X, Y):
        self.model.fit(X, Y)

    def predict(self, X):
        predicts = np.array([tree.predict(X) for tree in self.model.estimators_])
        
        mean = predicts.mean(axis=0)
        n_trees = len(self.model.estimators_)
        std = predicts.std(axis=0) / np.sqrt(n_trees)
        
        return mean, std
