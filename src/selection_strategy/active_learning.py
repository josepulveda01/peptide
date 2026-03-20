import numpy as np
from src.encoding.encoding import encode_batch

class UncertaintyStrategy:
    def __init__(self, sol_threshold=0.0):
        """
        Estrategia de active learning basada en incertidumbre.
        sol_threshold: umbral mínimo de solubilidad.
        """
        self.sol_threshold = sol_threshold

    def select(self, model, candidate_seqs, batch_size, encoding_method="physchem"):
        """
        Selecciona secuencias con mayor incertidumbre en la predicción de afinidad,
        considerando un umbral de solubilidad mínima.
        """
        # Codificación de secuencias
        X = encode_batch(candidate_seqs, method=encoding_method)

        # Predicción del modelo: mean y std
        mean, std = model.predict(X)
        std_aff = std[:, 0]        # desviación estándar de afinidad
        mean_sol = mean[:, 1]      # promedio de solubilidad

        # Filtrar candidatos por solubilidad
        mask = mean_sol >= self.sol_threshold
        scores = np.where(mask, std_aff, -np.inf)

        # Selección top-k (ordenados de mayor a menor incertidumbre)
        top_idx = np.argsort(scores)[-batch_size:][::-1]
        selected = [candidate_seqs[i] for i in top_idx]

        return selected
