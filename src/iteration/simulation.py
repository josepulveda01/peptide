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
from src.utilities.graphics import plot_experiment_history, plot_metrics, plot_strategy_comparison

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
    encoding_method="physchem"):
    
    sequences, aff, sol = generate_initial_data(init_size, length, seed=42)

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
        selected_candidates = strategy.select(model, candidates, batch_size, encoding_method=encoding_method)
        
        # Evaluación
        new_aff, new_sol = evaluate_sequences(selected_candidates, noisy=True)
        
        # Actualización del dataset
        sequences.extend(selected_candidates)
        aff = np.concatenate([aff, new_aff])
        sol = np.concatenate([sol, new_sol])
        
        # Tracking original
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
        
        
    return rounds_data, history_best_aff, history_mean_sol, metrics_history
    
        
# MAIN
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    
    strategies = {
        "Random" : RandomStrategy(),
        "UCB" : UCBStrategy(beta=1.0, sol_threshold=0.0),
        "Uncertainty" : UncertaintyStrategy(sol_threshold=0.0),
        "Evolutive" : EvolutiveStrategy(),
    }
    
    encoding_methods = ["one_hot", "physchem"]
    results = {}
    
    for name, strat in strategies.items():
        for method in encoding_methods:
            experiment_name = f"{name}_{method}" 
            print(f"Estrategia ejecutada: {experiment_name}")

            rounds_data, hist_aff, hist_sol, metrics_hist = run_simulation(
                strategy=strat,
                n_rounds=10,
                init_size=50,
                batch_size=10,
                length=5,
                encoding_method=method
            )
            
            plot_metrics(metrics_hist, experiment_name)
            
            # Guardar resultados en CSV
            result_dict = {
                "rounds": rounds_data,
                "metrics": metrics_hist
            }
            filepath = save_experiment(result_dict, seed=42, experiment_name=experiment_name)

            results[experiment_name] = {
                "history_best_aff": hist_aff,
                "history_mean_sol": hist_sol,
                "filepath": filepath
            }
            
    #plot_experiment_history(results, title_suffix="(OH vs PhysChem)")   
    plot_strategy_comparison(results, encoding_method="one_hot")
    plot_strategy_comparison(results, encoding_method="physchem")