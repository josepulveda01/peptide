# Optimización de secuencias peptídicas con machine learning


## 1. Descripción del proyecto

Este repositorio contiene un sistema para explorar de forma inteligente el espacio combinatorio de péptidos cortos (4–5 aminoácidos) con el objetivo de identificar candidatos con alta afinidad a una proteína diana (pIC50 > 7.0) y solubilidad aceptable (> 0.5 mg/mL).

Dado que la síntesis y medición experimental de péptidos es costosa y lenta, el proyecto simula un escenario realista mediante generación de datos sintéticos y aplica un ciclo iterativo de modelado predictivo y selección de secuencias inspirado en active learning y optimización bayesiana.

Se entrega como repositorio comprimido con código, datos generados y documentación de decisiones técnicas, listo para ejecución y presentación.


## 2. Funcionalidades principales

- Generación sintética de secuencias de péptidos y sus propiedades (afinidad y solubilidad).

- Codificación de secuencias en representaciones numéricas para modelado (one-hot o descriptor fisicoquímico).

- Modelos predictivos de propiedades de péptidos: Random Forest con estimación de incertidumbre.

- Estrategia de selección de nuevas secuencias usando Upper Confidence Bound (UCB).

- Ciclo iterativo de exploración: entrenar modelo → seleccionar candidatos → evaluar → reentrenar.

- Persistencia estructurada de datos por ronda (CSV).

- Visualizaciones de evolución de propiedades y trade-offs.

- Documentación del uso de LLMs en diseño y validación de prompts (docs/llm_usage.md).


## 3. Instalación

Se recomienda usar Python ≥ 3.10.

```python
git clone <repo_url>
cd peptide_project
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```


## 4. Uso

Ejemplo de un ciclo completo de exploración:

```python
from src.generator.generator import random_peptide_generator, evaluate_sequences
from src.models.random_forest import RandomForestWithUncertainty
from src.selection_strategy.ucb import UCBStrategy
from src.experiments.simulation import run_simulation_loop

# Generar conjunto inicial
initial_seqs = random_peptide_generator(n=20, length=5)
aff, sol = evaluate_sequences(initial_seqs, noisy=True, noise_std=0.1)

# Definir modelo y estrategia de selección
model = RandomForestWithUncertainty()
ucb = UCBStrategy(beta=1.0, sol_threshold=0.5)

# Ejecutar ciclo iterativo de exploración
results = run_simulation_loop(
    initial_seqs,
    aff,
    sol,
    model,
    ucb,
    n_rounds=5,     # Número de rondas
    batch_size=10,  # Secuencias nuevas por ronda
    save_path="data/results.csv"
)

```