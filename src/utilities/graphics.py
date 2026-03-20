import matplotlib.pyplot as plt
import numpy as np
import os

def plot_experiment_history(results, title_suffix=""):
    """
    Plotea métricas clave por ronda para múltiples experimentos.

    Parameters
    ----------
    results : dict
        Diccionario con claves de experimentos y valores de métricas por ronda.
        Ej:
        {
            "UCB_features": {
                "history_best_aff": [...],
                "history_mean_sol": [...],
                "history_mean_aff": [...],   # opcional
                "history_max_sol": [...]     # opcional
            },
            "UCB_one_hot": {...}
        }
    title_suffix : str
        Texto adicional para los títulos de los gráficos.
    """
    plt.figure(figsize=(15,5))

    # Mejor afinidad por ronda
    plt.subplot(1,3,1)
    for exp_name, result in results.items():
        plt.plot(result["history_best_aff"], label=exp_name)
    plt.xlabel("Ronda")
    plt.ylabel("Mejor afinidad")
    plt.title(f"Mejor afinidad {title_suffix}")
    plt.legend()

    # Afinidad promedio por ronda
    plt.subplot(1,3,2)
    for exp_name, result in results.items():
        # Si no existe, calculamos promedio a partir de history_best_aff (fallback)
        if "history_mean_aff" in result:
            y = result["history_mean_aff"]
        else:
            y = np.cumsum(result["history_best_aff"]) / np.arange(1, len(result["history_best_aff"])+1)
        plt.plot(y, label=exp_name)
    plt.xlabel("Ronda")
    plt.ylabel("Afinidad promedio")
    plt.title(f"Afinidad promedio {title_suffix}")
    plt.legend()

    # --- Solubilidad promedio por ronda ---
    plt.subplot(1,3,3)
    for exp_name, result in results.items():
        plt.plot(result["history_mean_sol"], label=exp_name)
    plt.xlabel("Ronda")
    plt.ylabel("Solubilidad promedio")
    plt.title(f"Solubilidad promedio {title_suffix}")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_metrics(metrics_history, experiment_name="experiment"):
    import os
    os.makedirs("results", exist_ok=True)
    
    rounds = [m["round"] for m in metrics_history]

    best_aff = [m["best_affinity"] for m in metrics_history]
    hit_rate = [m["hit_rate"] for m in metrics_history]
    rmse = [m["rmse"] for m in metrics_history]

    # --- Best affinity ---
    plt.figure()
    plt.plot(rounds, best_aff)
    plt.xlabel("Round")
    plt.ylabel("Best Affinity")
    plt.title(f"Best Affinity - {experiment_name}")
    plt.grid()
    plt.savefig(f"results/{experiment_name}_best_affinity.png")

    # --- Hit rate ---
    plt.figure()
    plt.plot(rounds, hit_rate)
    plt.xlabel("Round")
    plt.ylabel("Hit Rate")
    plt.title(f"Hit Rate - {experiment_name}")
    plt.grid()
    plt.savefig(f"results/{experiment_name}_hit_rate.png")

    # --- RMSE ---
    plt.figure()
    plt.plot(rounds, rmse)
    plt.xlabel("Round")
    plt.ylabel("RMSE")
    plt.title(f"RMSE - {experiment_name}")
    plt.grid()
    plt.savefig(f"results/{experiment_name}_rmse.png")

    plt.close("all")


def plot_strategy_comparison(results, encoding_method="physchem"):
    os.makedirs("results", exist_ok=True)

    plt.figure()

    for name, data in results.items():
        if encoding_method not in name:
            continue

        history = data["history_best_aff"]
        rounds = list(range(len(history)))

        plt.plot(rounds, history, label=name)

    plt.xlabel("Round")
    plt.ylabel("Best Affinity")
    plt.title(f"Strategy Comparison ({encoding_method})")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(f"results/comparison_{encoding_method}.png")
    plt.close()
    

def plot_with_confidence(all_results, encoding_method):
    """
    Grafica curvas promedio con intervalo de confianza (std) por estrategia.

    Parameters
    ----------
    all_results : dict
        {"strategy_name": [list of runs]}
        cada run = lista de métricas por ronda
    encoding_method : str
        "one_hot" o "physchem"
    """

    # Asegurar carpeta
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(8, 5))

    for strategy_name, runs in all_results.items():
        if len(runs) == 0:
            continue  # evitar crashes

        data = np.array(runs)  # shape (n_seeds, n_rounds)

        # Seguridad extra
        if data.ndim != 2:
            print(f"[WARNING] Forma inesperada en {strategy_name}: {data.shape}")
            continue

        mean = data.mean(axis=0)
        std = data.std(axis=0)

        x = np.arange(len(mean))

        # Línea principal
        plt.plot(x, mean, label=strategy_name)

        # Banda de incertidumbre
        plt.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.2
        )

    plt.title(f"Best Affinity vs Rounds ({encoding_method})")
    plt.xlabel("Rounds")
    plt.ylabel("Best Affinity")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"results/comparison_{encoding_method}.png")
    plt.close()

    
def plot_pareto(aff, sol, experiment_name="pareto"):
    os.makedirs("results", exist_ok=True)
    points = np.column_stack([aff, sol])
    pareto = []
    
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if all(q >= p) and any(q > p):
                dominated = True
                break
        if not dominated:
            pareto.append(p)

    pareto = np.array(pareto)

    plt.figure()
    plt.scatter(aff, sol, alpha=0.3, label="Samples")
    plt.scatter(pareto[:,0], pareto[:,1], label="Pareto", marker="x")

    plt.xlabel("Affinity")
    plt.ylabel("Solubility")
    plt.title(f"Pareto Frontier - {experiment_name}")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(f"results/pareto_{experiment_name}.png")
    plt.close()