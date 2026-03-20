import matplotlib.pyplot as plt
import numpy as np

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
