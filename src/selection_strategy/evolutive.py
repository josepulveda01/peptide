import numpy as np
import random
from src.generator.generator import aa_list
from src.encoding.encoding import encode_batch

def mutate_sequence(peptide, n_mutations=1):
    mutated_peptide = list(peptide)
    for _ in range(n_mutations):
        mutation_index = random.randint(0, len(peptide)-1)
        mutated_peptide[mutation_index] = random.choice(aa_list)
    return "".join(mutated_peptide)


class EvolutiveStrategy:
    """
    Estrategia evolutiva basada en mutaciones y selección de afinidad.
    """

    def __init__(self, mutation_fn=mutate_sequence, n_generations=5, n_offspring=50, sol_threshold=0.0):
        self.mutation_fn = mutation_fn
        self.n_generations = n_generations
        self.n_offspring = n_offspring
        self.sol_threshold = sol_threshold

    def select(self, model, candidate_seqs, batch_size, encoding_method="physchem"):
        """
        Selecciona secuencias usando un enfoque evolutivo con mutación y selección
        basada en afinidad, considerando un umbral mínimo de solubilidad.
        """
        population = candidate_seqs.copy()

        for _ in range(self.n_generations):
            offspring = [self.mutation_fn(seq) for seq in random.choices(population, k=self.n_offspring)]

            X_off = encode_batch(offspring, method=encoding_method)
            mean, std = model.predict(X_off)
            mean_aff = mean[:, 0]
            mean_sol = mean[:, 1]

            # Filtro por solubilidad
            mask = mean_sol >= self.sol_threshold
            fitness = np.where(mask, mean_aff, -np.inf)

            # Selección de los mejores (mantener tamaño constante)
            best_idx = np.argsort(fitness)[-len(population):]
            population = [offspring[i] for i in best_idx]

        # Finalmente, devolver top "batch_size"
        X_final = encode_batch(population, method=encoding_method)
        mean_final, _ = model.predict(X_final)
        mean_aff_final = mean_final[:, 0]

        top_idx = np.argsort(mean_aff_final)[-batch_size:]
        selected = [population[i] for i in top_idx]

        return selected
