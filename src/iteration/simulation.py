# src/experiments/simulation.py

import numpy as np
import random
import matplotlib.pyplot as plt
from src.generator.generator import(
    random_peptide_generator,
    affinity,
    solubility,
    evaluate_peptide,
    evaluate_sequences
)
from src.encoding.encoding import encode, encode_batch
from src.models.random_forest import RandomForestWithUncertainty
from src.selection_strategy.ucb import UCBStrategy

# evaluate_peptide() as lab_lecture()

# Data inicial SIN RUIDO
def generate_initial_data(N=50, lenght=5, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    sequences = [random_peptide_generator(lenght) for _ in range(N)]
    aff, sol = evaluate_sequences(sequences, noisy=False)
    return sequences, aff, sol


# SIMULACIÓN
def run_simulation(strategy, n_rounds=10, init_size=50, batch_size=10, length=5):
    sequences, aff, sol = generate_initial_data(init_size)
    history_best_aff = []
    history_mean_sol = []
    
    for r in range(n_rounds):
        print(f"Ronda {r}")
        
        # Encoding
        X = encode_batch(sequences)
        Y = np.column_stack([aff, sol])  # shape = (n_samples, 2)
        
        # Entrenamiento del RF
        model = RandomForestWithUncertainty()
        model.fit(X,Y)
        
        # 10M candidatos de largo L
        candidates = [random_peptide_generator(length) for _ in range(10 * batch_size)]

        
        # selección de candidatos
        selected = strategy.select(model, candidates, batch_size)
        
        # Evaluación
        new_aff, new_sol = evaluate_sequences(selected)
        
        # Actualización del dataset
        sequences.extend(selected)
        aff = np.concatenate([aff, new_aff])
        sol = np.concatenate([sol, new_sol])
        
        # Tracking
        best_aff = np.max(aff)
        mean_sol = np.mean(sol)
        history_best_aff.append(best_aff)
        history_mean_sol.append(mean_sol)
        
        print(f"Mejor afinidad hasta ahora: {best_aff:.3f}, Solubilidad media: {mean_sol:.3f}")
        
    return sequences, aff, sol, history_best_aff, history_mean_sol
        
        
# MAIN
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    
    strategies = {
        "UCB" : UCBStrategy(beta=1.0, sol_threshold=0.0)
    }
    
    results = {}
    
    for name, strat in strategies.items():
        print(f"Estrategia ejecutada: {name}")
        
        sequences, aff, sol, history_best_aff, history_mean_sol = (
            run_simulation(strategy=strat)
        )
        
        results[name] = {
            "history_best_aff": history_best_aff,
            "history_mean_sol": history_mean_sol
        }
    
    print(f"Ejecutando estrategia: UCB")
    
    np.random.seed(42)

    # -----------------------
    # Gráficos finales
    # -----------------------
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    for name, res in results.items():
        plt.plot(res["history_best_aff"], label=name)
    plt.xlabel("Round")
    plt.ylabel("Best affinity")
    plt.title("Evolución de afinidad")
    plt.legend()

    plt.subplot(1,2,2)
    for name, res in results.items():
        plt.plot(res["history_mean_sol"], label=name)
    plt.xlabel("Round")
    plt.ylabel("Mean solubility")
    plt.title("Evolución de solubilidad")
    plt.legend()

    plt.tight_layout()
    plt.show()