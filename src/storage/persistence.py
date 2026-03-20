from pathlib import Path
import csv
from datetime import datetime
import pandas as pd

# Carpeta donde se guardarán los resultados
DATA_FOLDER = Path(__file__).resolve().parent.parent.parent / "data"
DATA_FOLDER.mkdir(parents=True, exist_ok=True)  # crea la carpeta si no existe

def save_experiment(result, seed=None, experiment_name="experiment"):
    """
    Guarda los resultados de un experimento de simulación en un CSV con timestamp.
    
    Parameters
    ----------
    result : dict
        Diccionario con resultados por ronda. Debe contener:
        - "rounds": lista de dicts con métricas por ronda
    seed : int, optional
        Semilla usada para reproducibilidad
    experiment_name : str
        Nombre para el archivo (ej: 'UCB_features')
    """
    timestamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    filename = f"{experiment_name}_{timestamp}.csv"
    filepath = DATA_FOLDER / filename
    
    # Columnas del CSV
    fieldnames = [
        "round",
        "sequence",
        "mean_affinity",
        "max_affinity",
        "mean_solubility",
        "max_solubility",
        "seed"
    ]
    
    with filepath.open(mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for r, round_data in enumerate(result.get("rounds", [])):
            sequences = round_data.get("sequences", [])
            mean_aff = round_data.get("mean_affinity")
            max_aff = round_data.get("max_affinity")
            mean_sol = round_data.get("mean_solubility")
            max_sol = round_data.get("max_solubility")
            
            for seq in sequences:
                writer.writerow({
                    "round": r,
                    "sequence": seq,
                    "mean_affinity": mean_aff,
                    "max_affinity": max_aff,
                    "mean_solubility": mean_sol,
                    "max_solubility": max_sol,
                    "seed": seed
                })
    
    print(f"[INFO] Experimento guardado en {filepath}")
    return filepath


def data_load(csv_path):
    """
    Carga un CSV generado por save_experiment y devuelve un DataFrame.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo {csv_path}")

    df = pd.read_csv(csv_path)
    df['round'] = df['round'].astype(int)
    df['mean_affinity'] = df['mean_affinity'].astype(float)
    df['max_affinity'] = df['max_affinity'].astype(float)
    df['mean_solubility'] = df['mean_solubility'].astype(float)
    df['max_solubility'] = df['max_solubility'].astype(float)
    df['seed'] = df['seed'].astype(int)

    return df