import numpy as np
import random

from src.generator.generator import random_peptide_generator, evaluate_sequences
from src.encoding.encoding import encode_batch
from src.models.random_forest import RandomForestWithUncertainty

from src.selection_strategy.ucb import UCBStrategy
from src.selection_strategy.active_learning import UncertaintyStrategy
from src.selection_strategy.evolutive import EvolutiveStrategy
from src.selection_strategy.random_strategy import RandomStrategy

from src.evaluation.metrics import compute_metrics

from src.utilities.data_perstistence import save_experiment
from src.utilities.graphics import (
    plot_experiment_history, plot_metrics, plot_strategy_comparison, plot_with_confidence, plot_pareto)

# Data inicial SIN RUIDO
def generate_initial_data(N=50, length=5, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    sequences = [random_peptide_generator(length) for _ in range(N)]
    aff, sol = evaluate_sequences(sequences, noisy=False)
    return sequences, aff, sol


# SIMULACIÓN
def run_simulation(
    strategy,
    n_rounds=10,
    init_size=50,
    batch_size=10,
    length=5,
    encoding_method="physchem",
    seed=None
    ):
    
    sequences, aff, sol = generate_initial_data(init_size, length, seed=seed)

    # Mejores historicos
    history_best_aff = []
    history_mean_sol = []
    metrics_history = []
    rounds_data = []
    
    for r in range(n_rounds):
        print(f"Ronda {r}")
        
        # Encoding
        X = encode_batch(sequences, method=encoding_method)
        Y = np.column_stack([aff, sol])  # shape = (n_samples, 2)
        
        # Entrenamiento del RF
        model = RandomForestWithUncertainty()
        model.fit(X,Y)

        # Predicción 
        y_pred_mean, y_pred_std = model.predict(X)
        
        candidates = [random_peptide_generator(length) for _ in range(10 * batch_size)]
        selected_candidates = strategy.select(
            model,candidates, batch_size, encoding_method=encoding_method
        )
        
        # Evaluación
        new_aff, new_sol = evaluate_sequences(selected_candidates, noisy=True)
        
        # Actualización del dataset
        sequences.extend(selected_candidates)
        aff = np.concatenate([aff, new_aff])
        sol = np.concatenate([sol, new_sol])
        
        # Tracking 
        best_aff = np.max(aff)
        mean_sol = np.mean(sol)
        
        history_best_aff.append(best_aff)
        history_mean_sol.append(mean_sol)
        
        rounds_data.append({
           "round": r,
            "best_affinity": float(best_aff),
            "mean_solubility": float(mean_sol),
            "n_samples": len(aff)
        })

        # Metricas
        y_pred_aff = y_pred_mean[:, 0]
        metrics = compute_metrics(
            aff,
            sol,
            y_true=Y[:, 0],
            y_pred=y_pred_aff
        )
        metrics["round"] = r
        metrics_history.append(metrics)
    
        print(f"Mejor afinidad: {best_aff:.3f} | Hit rate: {metrics['hit_rate']:.3f} | RMSE: {metrics['rmse']}")
        
        
    return rounds_data, history_best_aff, history_mean_sol, metrics_history, aff, sol
    
        
# MAIN
if __name__ == "__main__":

    seeds = [0, 1, 2, 3, 4]
    strategies = {
        "Random": lambda: RandomStrategy(),
        "UCB": lambda: UCBStrategy(beta=1.0, sol_threshold=0.0),
        "Uncertainty": lambda: UncertaintyStrategy(sol_threshold=0.0),
        "Evolutive": lambda: EvolutiveStrategy(),
    }
    encoding_methods = ["one_hot", "physchem"]
    all_results = {
        "one_hot": {},
        "physchem": {}
    } # Para curvas con promedio

    for seed in seeds:
        print(f"Seed: {seed}")
        random.seed(seed)
        np.random.seed(seed)

        for name, strat in strategies.items():
            for method in encoding_methods:
                strategy = strat()
                experiment_key = f"{name}_{method}"
                experiment_name = f"{experiment_key}_seed{seed}"
                
                print(f"Estrategia: {experiment_name}")
                
                (rounds_data, hist_aff, hist_sol, metrics_hist, aff, sol ) = run_simulation(
                    strategy=strategy,
                    n_rounds=10,
                    init_size=50,
                    batch_size=10,
                    length=5,
                    encoding_method=method,
                    seed=seed
                )
                
                # Guardar para promedio
                if name not in all_results[method]:
                    all_results[method][name] = []

                all_results[method][name].append(hist_aff)

                # Métricas individuales
                plot_metrics(metrics_hist, experiment_name)

                # Pareto (última ronda)
                if seed == seeds[-1]:
                    plot_pareto(aff, sol, experiment_name)

                # Persistencia
                result_dict = {
                    "rounds": rounds_data,
                    "metrics": metrics_hist
                }

                save_experiment(
                    result_dict,
                    seed=seed,
                    experiment_name=experiment_name
                )
    plot_with_confidence(all_results["one_hot"], "one_hot")
    plot_with_confidence(all_results["physchem"], "physchem")
            
    #plot_experiment_history(results, title_suffix="(OH vs PhysChem)")   
    #plot_strategy_comparison(results, encoding_method="one_hot")
    #plot_strategy_comparison(results, encoding_method="physchem")