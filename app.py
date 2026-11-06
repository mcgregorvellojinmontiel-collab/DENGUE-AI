
import streamlit as st
import pandas as pd
import joblib

from pathlib import Path

# ==========================================
# 1. CONFIGURACION DE LA PAGINA
# ==========================================

st.set_page_config(
    page_title="DENGUE AI",
    page_icon="🦟",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

# ==========================================
# 2. CARGAR MODELOS
# ==========================================

@st.cache_resource
def cargar_modelos():

    modelo_logistico = joblib.load(
        BASE_DIR / "models" / "modelo_logistico.pkl"
    )

    modelo_neuronal = joblib.load(
        BASE_DIR / "models" / "modelo_red_neuronal.pkl"
    )

    return modelo_logistico, modelo_neuronal


try:
    modelo_logistico, modelo_neuronal = cargar_modelos()
except Exception as error:
    st.error(f"No se pudieron cargar los modelos: {error}")
    st.stop()

# ==========================================
# 3. ENCABEZADO
# ==========================================

st.title("🦟 DENGUE AI")

st.subheader(
    "Sistema de analítica predictiva "
    "para la clasificación del dengue"
)

st.caption(
    "Proyecto académico basado en regresión logística "
    "y redes neuronales artificiales."
)

st.divider()

# ==========================================
# 4. NAVEGACION
# ==========================================

opcion = st.sidebar.radio(
    "MENÚ PRINCIPAL",
    [
        "Inicio",
        "Predicción",
        "Comparación de modelos",
        "Información del proyecto"
    ]
)

# ==========================================
# 5. PAGINA DE INICIO
# ==========================================

if opcion == "Inicio":

    st.header("Panel principal")

    col1, col2, col3 = st.columns(3)

    col1.metric("Pacientes del dataset", "302")
    col2.metric("Modelos implementados", "2")
    col3.metric("Variables predictoras", "18")

    st.divider()

    st.subheader("Modelos disponibles")

    col1, col2 = st.columns(2)

    with col1:
        st.info("### Regresión logística")
        st.write(
            "Modelo de clasificación binaria "
            "que estima la probabilidad de dengue complicado."
        )

        st.metric("Accuracy", "85.25%")
        st.metric("AUC", "96.96%")

    with col2:
        st.success("### Red neuronal artificial")
        st.write(
            "Modelo de aprendizaje automático "
            "que identifica patrones mediante capas de neuronas."
        )

        st.metric("Accuracy", "77.05%")
        st.metric("AUC", "82.37%")

    st.divider()

    st.warning(
        "Esta aplicación es un prototipo académico. "
        "Sus predicciones no constituyen diagnósticos "
        "ni recomendaciones médicas."
    )

# ==========================================
# 6. PAGINA DE PREDICCION
# ==========================================

elif opcion == "Predicción":

    st.header("Predicción de dengue")

    st.write(
        "Ingrese los datos clínicos y de laboratorio "
        "para ejecutar uno o ambos modelos."
    )

    with st.form("formulario_paciente"):

        st.subheader("1. Datos demográficos")

        col1, col2 = st.columns(2)

        with col1:
            edad = st.selectbox(
                "Categoría de edad",
                [
                    "Menor de 6 años",
                    "Mayor de 6 años"
                ]
            )

        with col2:
            genero = st.selectbox(
                "Sexo",
                [
                    "Masculino",
                    "Femenino"
                ]
            )

        dias_fiebre = st.number_input(
            "Duración de la fiebre (días)",
            min_value=1,
            max_value=30,
            value=3
        )

        st.divider()

        st.subheader("2. Síntomas y evaluación clínica")

        col1, col2 = st.columns(2)

        sintomas = {}

        with col1:
            sintomas["HEADACHE"] = st.checkbox("Dolor de cabeza")
            sintomas["MYALGIA"] = st.checkbox("Dolor muscular")
            sintomas["ABDOMINAL PAIN"] = st.checkbox("Dolor abdominal")
            sintomas["RASH"] = st.checkbox("Erupciones en la piel")
            sintomas["VOMITING"] = st.checkbox("Vómitos")

        with col2:
            sintomas["BREATHLESSNESS"] = st.checkbox(
                "Dificultad respiratoria"
            )
            sintomas["BLEEDING"] = st.checkbox("Sangrado")
            sintomas["BP AT ADMISSION"] = st.checkbox(
                "Hipotensión al ingreso"
            )
            sintomas["ORGANOMEGALY"] = st.checkbox(
                "Organomegalia"
            )

        st.divider()

        st.subheader("3. Resultados de laboratorio")

        col1, col2 = st.columns(2)

        with col1:
            hb = st.number_input(
                "Hemoglobina (g/dL)",
                min_value=0.1,
                max_value=30.0,
                value=12.0,
                step=0.1
            )

            creatinina = st.number_input(
                "Creatinina (mg/dL)",
                min_value=0.01,
                max_value=20.0,
                value=0.5,
                step=0.01
            )

            pcv = st.number_input(
                "PCV (%)",
                min_value=0.0,
                max_value=100.0,
                value=40.0,
                step=0.1
            )

        with col2:
            plaquetas = st.number_input(
                "Recuento de plaquetas",
                min_value=0,
                max_value=2000000,
                value=150000,
                step=1000
            )

            wbc = st.number_input(
                "WBC máximo (células/cumm)",
                min_value=0,
                max_value=100000,
                value=6000,
                step=100
            )

            alt = st.number_input(
                "ALT (U/L)",
                min_value=0.0,
                max_value=10000.0,
                value=30.0,
                step=1.0
            )

        st.divider()

        st.subheader("4. Selección del modelo")

        seleccion = st.selectbox(
            "Modelo de inteligencia artificial",
            [
                "Regresión logística",
                "Red neuronal artificial",
                "Comparar ambos modelos"
            ]
        )

        ejecutar = st.form_submit_button(
            "Analizar paciente",
            type="primary",
            use_container_width=True
        )

    # ======================================
    # 7. PROCESAR PREDICCION
    # ======================================

    if ejecutar:

        datos = {
            "AGE": 0 if edad == "Menor de 6 años" else 1,
            "GENDER": 0 if genero == "Masculino" else 1,
            "DURATION OF FEVER": dias_fiebre,

            **sintomas,

            "HB": hb,
            "CREATININE": creatinina,

            "PCV_ALTO": int(pcv >= 42),
            "PLAQUETAS_BAJAS": int(plaquetas <= 75000),
            "WBC_ALTO": int(wbc >= 7000),
            "ALT_ALTO": int(alt >= 70)
        }

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

        entrada = pd.DataFrame(
            [[datos[c] for c in columnas]],
            columns=columnas
        )

        modelos_seleccionados = {}

        if seleccion == "Regresión logística":
            modelos_seleccionados = {
                "Regresión logística": modelo_logistico
            }

        elif seleccion == "Red neuronal artificial":
            modelos_seleccionados = {
                "Red neuronal artificial": modelo_neuronal
            }

        else:
            modelos_seleccionados = {
                "Regresión logística": modelo_logistico,
                "Red neuronal artificial": modelo_neuronal
            }

        st.divider()
        st.header("Resultados de la predicción")

        for nombre, modelo in modelos_seleccionados.items():

            prediccion = int(modelo.predict(entrada)[0])

            probabilidades = modelo.predict_proba(entrada)[0]

            probabilidad = float(probabilidades[1])

            st.subheader(nombre)

            if prediccion == 1:

                st.error("Clasificación: Dengue complicado")

            else:

                st.success("Clasificación: Dengue no complicado")

            col1, col2 = st.columns(2)

            col1.metric(
                "Probabilidad estimada de dengue complicado",
                f"{probabilidad * 100:.2f}%"
            )

            col2.metric(
                "Probabilidad estimada de dengue no complicado",
                f"{probabilidades[0] * 100:.2f}%"
            )

            st.progress(probabilidad)

        st.warning(
            "Resultado exclusivamente académico. "
            "No utilizar para diagnosticar, descartar complicaciones "
            "ni decidir tratamientos. La valoración clínica "
            "debe realizarla personal de salud."
        )

# ==========================================
# 8. COMPARACION DE MODELOS
# ==========================================

elif opcion == "Comparación de modelos":

    st.header("Comparación de rendimiento")

    resultados = pd.DataFrame({
        "Métrica": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "AUC"
        ],

        "Regresión logística": [
            0.852459,
            0.590909,
            1.000000,
            0.742857,
            0.969551
        ],

        "Red neuronal": [
            0.770492,
            0.476190,
            0.769231,
            0.588235,
            0.823718
        ]
    })

    st.dataframe(
        resultados,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Gráfico comparativo")

    grafico = resultados.set_index("Métrica")

    st.bar_chart(grafico)

    st.info(
        "En la evaluación inicial, la regresión logística "
        "obtuvo mejores resultados en las cinco métricas."
    )

    st.caption(
        "Los resultados corresponden a una única partición "
        "de 61 pacientes de prueba, no a una validación clínica."
    )

# ==========================================
# 9. INFORMACION DEL PROYECTO
# ==========================================

elif opcion == "Información del proyecto":

    st.header("Acerca de DENGUE AI")

    st.write(
        """
        DENGUE AI es un proyecto académico de analítica de datos
        que utiliza algoritmos de aprendizaje automático para
        clasificar registros de pacientes con dengue.

        El sistema implementa dos modelos:

        1. Regresión logística.
        2. Redes neuronales artificiales.

        Ambos modelos fueron entrenados utilizando una base
        de datos de 302 registros después del proceso de limpieza.
        """
    )

    st.subheader("Tecnologías utilizadas")

    st.write(
        """
        - Python
        - Pandas
        - NumPy
        - Scikit-learn
        - Streamlit
        - Matplotlib
        - Joblib
        """
    )

    st.subheader("Fuente científica")

    st.write(
        "Predictive scoring system for risk of complications "
        "in pediatric dengue infection (2022)."
    )

    st.caption(
        "Prototipo educativo. No validado para uso clínico."
    )

# ==========================================
# 10. PIE DE PAGINA
# ==========================================

st.divider()

st.caption(
    "DENGUE AI | Proyecto académico de analítica de datos"
)
