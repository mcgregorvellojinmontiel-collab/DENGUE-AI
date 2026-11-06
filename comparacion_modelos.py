
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

# ==========================================
# 1. CARGAR DATASET Y MODELOS
# ==========================================

df = pd.read_csv("data/datos_limpios.csv")

modelo_logistico = joblib.load(
    "models/modelo_logistico.pkl"
)

modelo_neuronal = joblib.load(
    "models/modelo_red_neuronal.pkl"
)

# ==========================================
# 2. DEFINIR VARIABLES
# ==========================================

columnas = [
    "AGE", "GENDER", "DURATION OF FEVER",
    "HEADACHE", "MYALGIA", "ABDOMINAL PAIN",
    "RASH", "VOMITING", "BREATHLESSNESS",
    "BLEEDING", "BP AT ADMISSION",
    "ORGANOMEGALY", "HB", "CREATININE",
    "PCV_ALTO", "PLAQUETAS_BAJAS",
    "WBC_ALTO", "ALT_ALTO"
]

objetivo = "GROUPS: DENGUE FEVER OR COMPLICATED DENGUE"

X = df[columnas]
y = df[objetivo]

# Misma particion de los entrenamientos anteriores
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

modelos = {
    "Regresion Logistica": modelo_logistico,
    "Red Neuronal": modelo_neuronal
}

# ==========================================
# 3. CALCULAR METRICAS
# ==========================================

resultados = []

for nombre, modelo in modelos.items():

    pred = modelo.predict(X_test)
    prob = modelo.predict_proba(X_test)[:, 1]

    resultados.append({
        "Modelo": nombre,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(
            y_test, pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, pred, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, pred, zero_division=0
        ),
        "AUC": roc_auc_score(y_test, prob)
    })

tabla = pd.DataFrame(resultados)

print("\n========== COMPARACION DE MODELOS ==========")
print(tabla.to_string(index=False))

# ==========================================
# 4. GUARDAR RESULTADOS
# ==========================================

Path("resultados").mkdir(exist_ok=True)

tabla.to_csv(
    "resultados/comparacion_metricas.csv",
    index=False
)

# ==========================================
# 5. GRAFICA DE METRICAS
# ==========================================

tabla_grafica = tabla.set_index("Modelo")

tabla_grafica.plot(
    kind="bar",
    figsize=(11, 6),
    ylim=(0, 1),
    rot=0
)

plt.title("Comparacion de modelos DENGUE AI")
plt.ylabel("Valor de la metrica")
plt.xlabel("Modelos")
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig(
    "resultados/comparacion_metricas.png",
    dpi=150
)

plt.close()

# ==========================================
# 6. MATRICES DE CONFUSION
# ==========================================

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for ax, (nombre, modelo) in zip(
    axes, modelos.items()
):
    ConfusionMatrixDisplay.from_estimator(
        modelo,
        X_test,
        y_test,
        display_labels=[
            "No complicado",
            "Complicado"
        ],
        cmap="Blues",
        ax=ax,
        colorbar=False
    )

    ax.set_title(nombre)

plt.tight_layout()

plt.savefig(
    "resultados/matrices_confusion.png",
    dpi=150
)

plt.close()

# ==========================================
# 7. CURVAS ROC
# ==========================================

fig, ax = plt.subplots(figsize=(7, 6))

for nombre, modelo in modelos.items():

    RocCurveDisplay.from_estimator(
        modelo,
        X_test,
        y_test,
        name=nombre,
        ax=ax
    )

ax.set_title("Curvas ROC - DENGUE AI")

plt.tight_layout()

plt.savefig(
    "resultados/curvas_roc.png",
    dpi=150
)

plt.close()

print("\nArchivos generados correctamente:")
print("- resultados/comparacion_metricas.csv")
print("- resultados/comparacion_metricas.png")
print("- resultados/matrices_confusion.png")
print("- resultados/curvas_roc.png")

print("\nCOMPARACION FINALIZADA")
