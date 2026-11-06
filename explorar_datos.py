
import pandas as pd
import numpy as np
from pathlib import Path

# =====================================
# 1. CARGAR LA BASE DE DATOS
# =====================================

ruta = Path("data/MASTER_CHART_F1000Research.xlsx")

df = pd.read_excel(ruta)

# Limpiar espacios en los nombres de columnas
df.columns = df.columns.str.strip()

print("\n=====================================")
print("     ANALISIS DE DATOS - DENGUE AI")
print("=====================================")

# =====================================
# 2. DIMENSIONES DE LA BASE DE DATOS
# =====================================

print("\n1. DIMENSIONES DE LA BASE DE DATOS")

print("Cantidad de pacientes:", df.shape[0])
print("Cantidad de variables:", df.shape[1])

# =====================================
# 3. NOMBRES DE LAS COLUMNAS
# =====================================

print("\n2. COLUMNAS DISPONIBLES")

for columna in df.columns:
    print("-", columna)

# =====================================
# 4. INFORMACION DE LAS VARIABLES
# =====================================

print("\n3. INFORMACION GENERAL")

df.info()

# =====================================
# 5. VALORES FALTANTES
# =====================================

print("\n4. VALORES FALTANTES")

faltantes = df.isnull().sum()

print(faltantes[faltantes > 0])

# =====================================
# 6. REGISTROS DUPLICADOS
# =====================================

print("\n5. REGISTROS DUPLICADOS")

duplicados = df.duplicated().sum()

print("Cantidad de duplicados:", duplicados)

# =====================================
# 7. COLUMNAS CONSTANTES
# =====================================

print("\n6. COLUMNAS CONSTANTES")

constantes = [
    columna for columna in df.columns
    if df[columna].nunique(dropna=False) <= 1
]

print(constantes if constantes else "No hay columnas constantes")

# =====================================
# 8. DISTRIBUCION DE LAS CLASES
# =====================================

print("\n7. DISTRIBUCION DE LOS PACIENTES")

objetivo = "GROUPS: DENGUE FEVER OR COMPLICATED DENGUE"

print(df[objetivo].value_counts())

# =====================================
# 9. ESTADISTICAS DESCRIPTIVAS
# =====================================

print("\n8. ESTADISTICAS DESCRIPTIVAS")

print(df.describe().round(2))

# =====================================
# 10. FINALIZACION
# =====================================

print("\n=====================================")
print(" ANALISIS COMPLETADO CORRECTAMENTE")
print("=====================================")
