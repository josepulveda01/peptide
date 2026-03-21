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

from src.iteration.simulation import run_simulation


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