
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# 1. CARGAR DATOS LIMPIOS
# ==========================================

df = pd.read_csv("data/datos_limpios.csv")

print("\n===== REGRESION LOGISTICA - DENGUE AI =====")

# ==========================================
# 2. DEFINIR VARIABLES PREDICTORAS
# ==========================================

columnas = [
    "AGE",
    "GENDER",
    "DURATION OF FEVER",
    "HEADACHE",
    "MYALGIA",
    "ABDOMINAL PAIN",
    "RASH",
    "VOMITING",
    "BREATHLESSNESS",
    "BLEEDING",
    "BP AT ADMISSION",
    "ORGANOMEGALY",
    "HB",
    "CREATININE",
    "PCV_ALTO",
    "PLAQUETAS_BAJAS",
    "WBC_ALTO",
    "ALT_ALTO"
]

objetivo = "GROUPS: DENGUE FEVER OR COMPLICATED DENGUE"

X = df[columnas]
y = df[objetivo]

print("\nCantidad de variables predictoras:", X.shape[1])
print("Cantidad de pacientes:", X.shape[0])

# ==========================================
# 3. DIVIDIR DATOS DE ENTRENAMIENTO Y PRUEBA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nDatos de entrenamiento:", len(X_train))
print("Datos de prueba:", len(X_test))

# ==========================================
# 4. CREAR EL MODELO
# ==========================================

modelo = Pipeline([
    ("escalador", StandardScaler()),

    ("regresion", LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42
    ))
])

# ==========================================
# 5. ENTRENAR EL MODELO
# ==========================================

print("\nEntrenando modelo...")

modelo.fit(X_train, y_train)

print("Entrenamiento completado.")

# ==========================================
# 6. REALIZAR PREDICCIONES
# ==========================================

predicciones = modelo.predict(X_test)

probabilidades = modelo.predict_proba(X_test)[:, 1]

# ==========================================
# 7. EVALUAR EL MODELO
# ==========================================

accuracy = accuracy_score(y_test, predicciones)

precision = precision_score(
    y_test, predicciones, zero_division=0
)

sensibilidad = recall_score(
    y_test, predicciones, zero_division=0
)

f1 = f1_score(
    y_test, predicciones, zero_division=0
)

auc = roc_auc_score(y_test, probabilidades)

print("\n========== RESULTADOS ==========")

print(f"Exactitud (Accuracy): {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Sensibilidad (Recall): {sensibilidad:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")

# ==========================================
# 8. MATRIZ DE CONFUSION
# ==========================================

print("\nMATRIZ DE CONFUSION")

matriz = confusion_matrix(
    y_test,
    predicciones,
    labels=[0, 1]
)

print(matriz)

# ==========================================
# 9. REPORTE DE CLASIFICACION
# ==========================================

print("\nREPORTE DE CLASIFICACION")

print(
    classification_report(
        y_test,
        predicciones,
        target_names=[
            "Dengue no complicado",
            "Dengue complicado"
        ],
        zero_division=0
    )
)

# ==========================================
# 10. GUARDAR MODELO ENTRENADO
# ==========================================

Path("models").mkdir(exist_ok=True)

joblib.dump(
    modelo,
    "models/modelo_logistico.pkl"
)

print("\nModelo guardado correctamente.")

print("Archivo: models/modelo_logistico.pkl")

print("\n===== PROCESO FINALIZADO =====")
