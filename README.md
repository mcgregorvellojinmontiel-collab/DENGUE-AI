
# DENGUE AI

Sistema académico de analítica predictiva para
la clasificación de dengue no complicado y complicado.

## Modelos implementados

1. Regresión logística.
2. Redes neuronales artificiales.

## Dataset

Base de datos de pacientes con dengue,
utilizada para entrenar los modelos.

## Tecnologías

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

## Resultados iniciales

| Métrica | Regresión logística | Red neuronal |
|---|---:|---:|
| Accuracy | 85.25% | 77.05% |
| Precision | 59.09% | 47.62% |
| Recall | 100% | 76.92% |
| F1-score | 74.29% | 58.82% |
| AUC | 96.96% | 82.37% |

## Ejecución local

Instalar dependencias:

pip install -r requirements.txt

Ejecutar:

streamlit run app.py

## Advertencia

Proyecto académico, no destinado al diagnóstico
ni a la toma de decisiones clínicas.
