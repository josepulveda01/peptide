# Generador de secuencias y funciones de solubilidad/afinidad

import numpy as np
import pandas as pd
import random 

# hidrofoficidad (h), carga (q) y volumen (V) para cada aminoácido
aminoacid_features = {
    "A": (1.8, 0,  88),
    "R": (-4.5, 1, 173),
    "N": (-3.5, 0, 114),
    "D": (-3.5,-1, 111),
    "C": (2.5, 0, 108),
    "Q": (-3.5, 0, 143),
    "E": (-3.5,-1, 138),
    "G": (-0.4, 0,  60),
    "H": (-3.2, 0, 153),
    "I": (4.5, 0, 166),
    "L": (3.8, 0, 166),
    "K": (-3.9, 1, 168),
    "M": (1.9, 0, 162),
    "F": (2.8, 0, 189),
    "P": (-1.6, 0, 112),
    "S": (-0.8, 0,  89),
    "T": (-0.7, 0, 116),
    "W": (-0.9, 0, 227),
    "Y": (-1.3, 0, 193),
    "V": (4.2, 0, 140),
}

aa_list = list(aminoacid_features.keys())


def random_peptide_generator(length=5):
    return "".join(random.choices(aa_list, k=length))


def peptide_features(peptide):
    h = np.array([aminoacid_features[aa][0] for aa in peptide]) # hidrofobicidad
    q = np.array([aminoacid_features[aa][1] for aa in peptide]) # carga
    v = np.array([aminoacid_features[aa][2] for aa in peptide]) # volumen
    return h, q, v


# solubilidad intrínseca del peptido
def solubility(peptide): # MODELO MEJORABLE
    h, q, _ = peptide_features(peptide)
    
    total_hydro = np.sum(h)
    local_hydro_interaction = np.sum( h[:-1]*h[1:] )
    charge_effect = np.sum(np.abs(q))
    
    return -0.8*total_hydro -0.2*local_hydro_interaction + 1.5*charge_effect


# afinidad intrínseca del peptido
def affinity(peptide):
    _, _, v = peptide_features(peptide)
    
    total_volume = 0.5 * np.sum(v) / 100                     # scaling para evitar dominio artificial
    local_interaction = 0.3 * np.sum(v[:-1] * v[1:]) / 10000 # scaling para evitar dominio artificial
    
    # bonus por motiv - incluir motiv estratégicos
    motif_bonus = 0
    if "WF" in peptide:
        motif_bonus += 2

    return total_volume + local_interaction + motif_bonus


# simula una medición experimental de solubilidad y afinidad, continene ruido
def evaluate_peptide(peptide, noisy=True, noise_std=0.1):
    sol, aff = solubility(peptide), affinity(peptide)
    if noisy:
        sol += np.random.normal(0, noise_std)
        aff += np.random.normal(0, noise_std)
    return sol, aff


def generate_dataset(N=200, length=5, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    data = []
    for _ in range(N):
        p = random_peptide_generator(length)
        data.append((p, solubility(p), affinity(p)))

    return pd.DataFrame(data, columns=["sequence", "solubility", "affinity"])

if __name__ == "__main__":
    # Generar dataset inicial de ejemplo
    df = generate_dataset(N=200, length=5, seed=42)
    print(df.head())
    df.to_csv("data/initial_dataset.csv", index=False) # cambiar directorio

