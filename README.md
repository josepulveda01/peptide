# Optimización de secuencias peptídicas con _machine learning_
> Este proyecto aborda el desafío de explorar eficientemente un espacio combinatorio extremadamente grande de secuencias peptídicas, donde la evaluación experimental es costosa y limitada.

## Tabla de contenidos
* [Información general](#información-general)
* [Enfoque](#enfoque)
* [Modelo](#modelo)
* [Estrategia de selección](#estrategia-de-selección)
* [Estructura del proyecto](#estructura-del-proyecto)
* [Implementación](#implementación)
* [Uso](#uso)
* [Limitaciones](#limitaciones)
* [Futuras implementaciones](#futuras-implementaciones)
* [Declaraciones de LLM](#declaraciones-de-llm)
* [Repositorio en GitHub](#repositorio)
* [Contacto](#contacto)
<!-- * [License](#license) -->


## Información general

La síntesis de péptidos es un proceso experimental altamente costoso y lento. El espacio muestral de una cadena de $n$ aminoácidos es del orden de $20^n$ elemento, lo que limita enormemente su viabilidad experimental para cadenas largas.

En ese contexto, nace la necesidad de buscar mecanismos de exploración que permitan recorrer este espacio de forma eficiente y, a partir de la menor cantidad de observaciones posible, maximizar la variable experimental deseada.

En este problema particular, se busca optimizar simultáneamente:
- Afinidad de unión a una proteína diana
- Solubilidad en medio acuoso

Y dado que la evaluación experimental puede considerarse una función de caja negra, este problema puede reinterpretarse como uno de **optimización bayesiana**.





## Enfoque

Se propone un esquema de aprendizaje iterativo basado en el siguiente ciclo:

Datos → Codificación → Entrenamiento → Selección → Evaluación → Datos

En cada iteración:
1. Se entrena un modelo predictivo
2. Se generan nuevas secuencias candidatas
3. Se seleccionan las más prometedoras
4. Se evalúan (mediante una función sintética)
5. Se incorporan al dataset



## Modelo

Se utiliza un modelo de **Random Forest** para predecir:
- Afinidad (pIC50)
- Solubilidad

Además, se emplean dos estrategias de representación:
- **_One-hot encoding_** (agnóstico)
- **_Features_ fisicoquímicas** (informadas)

Esto permite comparar el impacto de incorporar conocimiento del dominio en el proceso de aprendizaje. El modelo también permite estimar incertidumbre mediante la variabilidad entre árboles, lo cual es clave para las estrategias de selección.


## Estrategia de selección

Se implementan tres enfoques principales:

- **Función de adquisición (UCB)**  
Balancea exploración y explotación

- **Aprendizaje activo (_uncertainty sampling_)**  
  Prioriza regiones con alta incertidumbre

- **Estrategia evolutiva**  
  Genera nuevas secuencias mediante mutaciones

El objetivo es analizar cómo estas estrategias afectan el desempeño del sistema.



## Estructura del proyecto

Arquitectura modular orientada a experimentación iterativa.

    .
    ├── data/                    # Almacenamiento de datos
    ├── results/                 # Resultados de simulaciones 
    ├── src/                     # Código fuente principal
    │   ├── encoding/            # Codificación numérica de peptidos
    │   ├── evaluation/          # Métricas
    │   ├── generator/           # Generador de péptidos
    │   ├── iteration/           # Loop
    │   ├── models/              # Modelos de ML
    │   ├── selection_strategy/  # Estrategias de selección de candidatos
    │   └── utilities/           # Funciones auxiliares
    ├── main.py                  # Script principal para ejecutar experimentos
    ├── config.py                # Configuración de parámetros
    ├── requirements.txt         # Dependencias del proyecto
    └── README.md                # Documentación



## Implementación

```bash
pip install -r requirements.txt
python main.py
```

## Uso

Para ejecutar una simulación básica:

```bash
python -m main.py
```



## Limitaciones

Debido al plazo acotado para el proyecto, aún se encuentra en desarrollo. A fecha de 20-03-2026, se identifican las siguientes limitaciones:

- Uso de datos sintéticos
- Falta de validación formal del modelo
- Optimización multiobjetivo simplificada
- Exploración del espacio limitada
- Escalabilidad restringida
- Modelo relativamente simple
- Incertidumbre del _Random Forest_ aproximada
- Dependencia del diseño de la función sintética



## Futuras implementaciones

En futuras iteraciones del proyecto, se espera poder cubrir la mayoría de limitaciones. Principalmente:

- Métodos de validación: _train/test split_ y _cross validation_
- Optimización de Pareto
- Optimización Bayesiana con procesos gaussianos
- Optimización del código para escalabilidad a secuencias más largas
- Mejora del sistema de almacenamiento mediante SQL
- Parámetros inciales en un archivo config.py


## Declaraciones de LLM

Se utilizó ChatGPT como herramienta de apoyo para:

- Contextualización del problema
- Estructuración del código
- Diligencia de labores simples
- Asistencia en debugging
- Detección de _typos_

Todas las decisiones de diseño, validación de resultados e implementación final fueron realizadas y verificadas manualmente por el autor.



## Repositorio de GitHub

Este repositorio se encuentra subido a GitHub. Puede consultar detalles sobre commits, evolución y estado actual del proyecto en el siguiente enlace:
https://github.com/josepulveda01/peptide.git

## Contacto

**Autor:** José Sepúlveda \
**Correo:** josepulveda01@hotmail.com