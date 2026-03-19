# BAYESIAN OPTIMIZATION

import numpy as np
from src.encoding import encode_batch

class UCBStrategy:
    def __init__(self, beta=1.0, sol_threshold=0.0):
        self.beta = beta
        self.sol_threshold = sol_threshold

    def select(self, model, candidate_seqs, batch_size):
        """
        Selecciona secuencias usando Upper Confidence Bound (UCB),
        considerando un umbral de solubilidad mínima.
        """
        X = encode_batch(candidate_seqs)
        mean, std = model.predict(X)
        
        mean_aff = mean[:, 0]
        mean_sol = mean[:, 1]
        std_aff  = std[:, 0]
        
        # UCB sobre afinidad
        ucb_score = mean_aff + self.beta * std_aff

        # filtro por solubilidad
        mask = mean_sol >= self.sol_threshold
        scores = np.where(mask, ucb_score, -np.inf)

        # selección top-k (ordenados de mejor a peor)
        top_idx = np.argsort(scores)[-batch_size:][::-1]
        selected = [candidate_seqs[i] for i in top_idx]

        return selected
