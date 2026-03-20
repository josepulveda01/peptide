import numpy as np
from src.generator.generator import aminoacid_features, aa_list

aa_to_index =  { aa:i for i,aa in enumerate(aa_list)}

def encode(peptide, method="physchem"):
    # one-hot encoding: 20L variables
    if method == "one_hot":
        L = len(peptide)
        encoding = np.zeros((L,len(aa_list))) # Lx20
    
        for i,aa in enumerate(peptide):
            aa_index = aa_to_index[aa]
            encoding[i,aa_index] = 1
    
        return encoding.flatten()
    
    # physicochemical properties: 3L+3+3 = 3(L+2) variables
    elif method == "physchem":
        h = np.array([ aminoacid_features[aa][0] for aa in peptide ])
        q = np.array([ aminoacid_features[aa][1] for aa in peptide ])
        v = np.array([ aminoacid_features[aa][2] for aa in peptide ])
        
        positional_features = np.concatenate([h,q,v])
        global_features = np.array( [np.sum(h), np.sum(q), np.sum(v)] )
        local_interactions = np.array(
            [np.sum(h[:-1]*h[1:]),
            np.sum(q[:-1]*q[1:]),
            np.sum(v[:-1]*v[1:])]
        )

        return np.concatenate([positional_features, global_features, local_interactions])
    
    else:
        raise ValueError(f"Método de encoding desconocido: {method}")

# encoding de listas
def encode_batch(sequences, method="physchem"):
    return np.vstack([encode(peptide, method) for peptide in sequences])
        

def encode_dataset(df, method="physchem"):
    #return np.vstack([encode(seq,method) for seq in df["sequence"]])
    return encode_batch(df["sequence"], method)