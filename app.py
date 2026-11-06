
import streamlit as st
import pandas as pd
import joblib

from pathlib import Path

# =====================================================
# 1. CONFIGURACION GENERAL
# =====================================================

st.set_page_config(
    page_title="DENGUE AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

# =====================================================
# 2. ESTILOS VISUALES
# =====================================================

st.markdown("""
<style>

/* =========================
   ESTILO GENERAL
========================= */

html, body, .stApp {
    background-color: #FFFFFF !important;
    color: #263746 !important;
    font-family: Arial, sans-serif !important;
}

[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stHeader"],
[data-testid="stMainBlockContainer"] {
    background-color: #FFFFFF !important;
}

/* =========================
   SIDEBAR
========================= */

[data-testid="stSidebar"] {
    background-color: #F5F7F9 !important;
    border-right: 1px solid #E1E6EB !important;
}

[data-testid="stSidebar"] * {
    color: #263746 !important;
}

/* =========================
   TÍTULOS Y TEXTO
========================= */

h1, h2, h3, h4 {
    color: #193650 !important;
    font-weight: 600 !important;
}

p, label, span, div {
    color: #34495E !important;
}

/* =========================
   FORMULARIOS
========================= */

[data-testid="stForm"] {
    background-color: #FFFFFF !important;
    border: none !important;
    padding: 0 !important;
}

/* Inputs numéricos y texto */
input, textarea {
    background-color: #FFFFFF !important;
    color: #263746 !important;
    border: 1px solid #D5DCE3 !important;
    border-radius: 4px !important;
}

/* =========================
   SELECTBOX - CAJA PRINCIPAL
========================= */

/* Caja visible del select */
div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: #263746 !important;
    border: 1px solid #D5DCE3 !important;
    border-radius: 4px !important;
    box-shadow: none !important;
}

/* Texto dentro del select */
div[data-baseweb="select"] span {
    color: #263746 !important;
}

/* Flecha / icono del select */
div[data-baseweb="select"] svg {
    fill: #263746 !important;
    color: #263746 !important;
}

/* Zona derecha del select donde sale oscuro */
div[data-baseweb="select"] div[role="button"] {
    background-color: #FFFFFF !important;
    color: #263746 !important;
}

/* =========================
   MENÚ DESPLEGABLE DEL SELECT
========================= */

/* Popover general */
[data-baseweb="popover"] {
    background-color: #FFFFFF !important;
    color: #263746 !important;
}

/* Menú lista */
ul[role="listbox"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D5DCE3 !important;
    color: #263746 !important;
}

/* Opciones */
ul[role="listbox"] li {
    background-color: #FFFFFF !important;
    color: #263746 !important;
}

/* Hover de opciones */
ul[role="listbox"] li:hover {
    background-color: #F1F5F9 !important;
    color: #263746 !important;
}

/* Opción seleccionada o activa */
[aria-selected="true"] {
    background-color: #EAF1F7 !important;
    color: #193650 !important;
}

/* A veces Streamlit usa role option */
div[role="option"] {
    background-color: #FFFFFF !important;
    color: #263746 !important;
}

div[role="option"]:hover {
    background-color: #F1F5F9 !important;
    color: #263746 !important;
}

/* =========================
   CHECKBOX
========================= */

[data-testid="stCheckbox"] label {
    color: #263746 !important;
}

[data-testid="stCheckbox"] div[role="checkbox"] {
    background-color: #FFFFFF !important;
    border: 1px solid #C7D0D9 !important;
}

[data-testid="stCheckbox"] svg {
    fill: #244B70 !important;
    color: #244B70 !important;
}

/* =========================
   MÉTRICAS
========================= */

[data-testid="stMetric"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 8px 0 !important;
}

[data-testid="stMetricLabel"] {
    color: #718096 !important;
}

[data-testid="stMetricValue"] {
    color: #193650 !important;
    font-weight: 600 !important;
}


/* BOTONES */

.stButton > button,
.stFormSubmitButton > button,
[data-testid="stFormSubmitButton"] button {
    background-color: #244B70 !important;
    color: #FFFFFF !important;
    border: 1px solid #244B70 !important;
    border-radius: 4px !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

/* Forzar texto blanco dentro del botón */

.stButton > button *,
.stFormSubmitButton > button *,
[data-testid="stFormSubmitButton"] button * {
    color: #FFFFFF !important;
    opacity: 1 !important;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background-color: #193650 !important;
    border-color: #193650 !important;
}


/* =========================
   ALERTAS Y PROGRESO
========================= */

[data-testid="stAlert"] {
    border-radius: 4px !important;
}

[data-testid="stProgressBar"] > div > div {
    background-color: #244B70 !important;
}

/* =========================
   DIVISORES Y TABLAS
========================= */

hr {
    border: none !important;
    border-top: 1px solid #E1E6EB !important;
}

[data-testid="stDataFrame"] {
    border: none !important;
}

/* =========================
   OCULTAR ELEMENTOS
========================= */


/* AVISO DE USO ACADEMICO */

.disclaimer {
    background-color: #F2F5F8 !important;
    color: #34495E !important;
    border-left: 4px solid #244B70;
    padding: 16px 20px;
    margin-top: 20px;
    margin-bottom: 15px;
    border-radius: 3px;
    font-size: 14px;
    line-height: 1.7;
}

.disclaimer strong {
    color: #193650 !important;
    font-weight: 600;
}


#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. CARGAR LOS MODELOS
# =====================================================

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

    st.error(
        "No fue posible cargar los modelos. "
        "Verifique los archivos de la carpeta models."
    )

    st.exception(error)

    st.stop()


# =====================================================
# 4. VARIABLES DEL PROYECTO
# =====================================================

COLUMNAS = [
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

METRICAS = pd.DataFrame({
    "Métrica": [
        "Exactitud",
        "Precisión",
        "Sensibilidad",
        "Puntuación F1",
        "Área bajo la curva ROC"
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


# =====================================================
# 5. ENCABEZADO DE LA APLICACION
# =====================================================

st.sidebar.markdown(
    "# DENGUE AI"
)

st.sidebar.caption(
    "PLATAFORMA DE ANALÍTICA DE DATOS"
)

st.sidebar.divider()


# =====================================================
# 6. MENU DE NAVEGACION
# =====================================================

pagina = st.sidebar.radio(
    "NAVEGACIÓN",
    [
        "Inicio",
        "Predicción",
        "Rendimiento de modelos",
        "Información del proyecto"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Proyecto académico"
)

st.sidebar.caption(
    "Aprendizaje automático · Python"
)


# =====================================================
# 7. PAGINA DE INICIO
# =====================================================

if pagina == "Inicio":

    st.caption(
        "PLATAFORMA ACADÉMICA DE ANALÍTICA DE DATOS"
    )

    st.title("DENGUE AI")

    st.write(
        "Sistema de clasificación predictiva de dengue "
        "mediante técnicas de aprendizaje automático."
    )

    st.divider()

    st.subheader("Resumen del proyecto")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Registros analizados",
        "302"
    )

    col2.metric(
        "Modelos desarrollados",
        "2"
    )

    col3.metric(
        "Variables predictoras",
        "18"
    )

    st.divider()

    st.subheader("Modelos implementados")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("**Regresión logística**")

        st.write(
            "Modelo estadístico de clasificación binaria "
            "que estima la probabilidad de pertenecer "
            "a la categoría de dengue complicado."
        )

        st.metric(
            "Exactitud",
            "85,25 %"
        )

        st.write("Área bajo la curva ROC: 96,96 %")

    with col2:

        st.markdown("**Red neuronal artificial**")

        st.write(
            "Modelo de aprendizaje automático que "
            "identifica patrones mediante una arquitectura "
            "de capas neuronales."
        )

        st.metric(
            "Exactitud",
            "77,05 %"
        )

        st.write("Área bajo la curva ROC: 82,37 %")

    st.divider()

    st.subheader("Objetivo del sistema")

    st.write(
        "Desarrollar una plataforma que permita analizar "
        "datos clínicos y de laboratorio de pacientes "
        "con dengue, utilizando dos modelos de clasificación "
        "para comparar sus resultados."
    )

    st.write(
        "El proyecto contempla la preparación de datos, "
        "el entrenamiento de modelos de aprendizaje automático "
        "y la evaluación de su rendimiento."
    )

    st.divider()

    st.caption(
        "Nota: herramienta de uso académico. "
        "Los resultados no constituyen un diagnóstico "
        "ni sustituyen la valoración de un profesional de salud."
    )


# =====================================================
# 8. PAGINA DE PREDICCION
# =====================================================

elif pagina == "Predicción":

    st.title("Predicción")

    st.write(
        "Ingrese la información del paciente para generar "
        "una clasificación mediante los modelos entrenados."
    )

    st.divider()

    with st.form("formulario_paciente"):

        # ---------------------------------------------
        # DATOS DEMOGRAFICOS
        # ---------------------------------------------

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

        # ---------------------------------------------
        # SINTOMAS
        # ---------------------------------------------

        st.subheader("2. Síntomas y evaluación clínica")

        col1, col2 = st.columns(2)

        with col1:

            headache = st.checkbox(
                "Dolor de cabeza"
            )

            myalgia = st.checkbox(
                "Dolor muscular"
            )

            abdominal_pain = st.checkbox(
                "Dolor abdominal"
            )

            rash = st.checkbox(
                "Erupciones cutáneas"
            )

            vomiting = st.checkbox(
                "Vómitos"
            )

        with col2:

            breathlessness = st.checkbox(
                "Dificultad respiratoria"
            )

            bleeding = st.checkbox(
                "Sangrado"
            )

            hypotension = st.checkbox(
                "Hipotensión al ingreso"
            )

            organomegaly = st.checkbox(
                "Organomegalia"
            )

        st.divider()

        # ---------------------------------------------
        # LABORATORIOS
        # ---------------------------------------------

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
                value=0.50,
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

        # ---------------------------------------------
        # SELECCION DEL MODELO
        # ---------------------------------------------

        st.subheader("4. Selección del modelo")

        seleccion = st.selectbox(
            "Modelo de clasificación",
            [
                "Regresión logística",
                "Red neuronal artificial",
                "Comparar ambos modelos"
            ]
        )

        ejecutar = st.form_submit_button(
            "Ejecutar predicción",
            use_container_width=True
        )

    # =================================================
    # PROCESAMIENTO DE LA PREDICCION
    # =================================================

    if ejecutar:

        # Codificación de los datos

        datos = {

            "AGE": 0 if edad == "Menor de 6 años" else 1,

            "GENDER": 0 if genero == "Masculino" else 1,

            "DURATION OF FEVER": dias_fiebre,

            "HEADACHE": int(headache),

            "MYALGIA": int(myalgia),

            "ABDOMINAL PAIN": int(abdominal_pain),

            "RASH": int(rash),

            "VOMITING": int(vomiting),

            "BREATHLESSNESS": int(breathlessness),

            "BLEEDING": int(bleeding),

            "BP AT ADMISSION": int(hypotension),

            "ORGANOMEGALY": int(organomegaly),

            "HB": hb,

            "CREATININE": creatinina,

            "PCV_ALTO": int(pcv >= 42),

            "PLAQUETAS_BAJAS": int(plaquetas <= 75000),

            "WBC_ALTO": int(wbc >= 7000),

            "ALT_ALTO": int(alt >= 70)
        }

        entrada = pd.DataFrame(
            [[datos[columna] for columna in COLUMNAS]],
            columns=COLUMNAS
        )

        # Seleccionar modelos

        if seleccion == "Regresión logística":

            modelos = {
                "Regresión logística": modelo_logistico
            }

        elif seleccion == "Red neuronal artificial":

            modelos = {
                "Red neuronal artificial": modelo_neuronal
            }

        else:

            modelos = {
                "Regresión logística": modelo_logistico,
                "Red neuronal artificial": modelo_neuronal
            }

        # Mostrar resultados

        st.divider()

        st.subheader("Resultados de la predicción")

        for nombre, modelo in modelos.items():

            prediccion = int(
                modelo.predict(entrada)[0]
            )

            probabilidades = modelo.predict_proba(
                entrada
            )[0]

            prob_no_complicado = float(
                probabilidades[0]
            )

            prob_complicado = float(
                probabilidades[1]
            )

            st.markdown(f"### {nombre}")

            if prediccion == 1:

                st.write(
                    "**Clasificación del modelo:** "
                    "Dengue complicado"
                )

            else:

                st.write(
                    "**Clasificación del modelo:** "
                    "Dengue no complicado"
                )

            col1, col2 = st.columns(2)

            col1.metric(
                "Probabilidad estimada de dengue no complicado",
                f"{prob_no_complicado * 100:.2f} %"
            )

            col2.metric(
                "Probabilidad estimada de dengue complicado",
                f"{prob_complicado * 100:.2f} %"
            )

            st.progress(prob_complicado)

            st.divider()

        st.markdown("""
        <div class="disclaimer">

            <strong>Advertencia:</strong>

            Estas predicciones son resultados experimentales
            de modelos académicos. No deben utilizarse para
            diagnosticar, descartar complicaciones ni decidir
            tratamientos médicos.

        </div>
        """, unsafe_allow_html=True)



# =====================================================
# 9. RENDIMIENTO DE LOS MODELOS
# =====================================================

elif pagina == "Rendimiento de modelos":

    st.title("Rendimiento de modelos")

    st.write(
        "Comparación de los resultados obtenidos "
        "por los dos algoritmos durante la evaluación."
    )

    st.divider()

    st.subheader("Comparación de métricas")

    tabla = METRICAS.copy()

    tabla["Regresión logística"] = (
        tabla["Regresión logística"] * 100
    ).map(lambda x: f"{x:.2f} %")

    tabla["Red neuronal"] = (
        tabla["Red neuronal"] * 100
    ).map(lambda x: f"{x:.2f} %")

    st.dataframe(
        tabla,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Representación gráfica")

    grafico = METRICAS.set_index("Métrica")

    st.bar_chart(
        grafico,
        y=[
            "Regresión logística",
            "Red neuronal"
        ]
    )

    st.divider()

    st.subheader("Interpretación de los resultados")

    st.write(
        "En la evaluación inicial, la regresión logística "
        "obtuvo mejores resultados que la red neuronal "
        "en las cinco métricas analizadas."
    )

    st.write(
        "**Regresión logística:** obtuvo una exactitud "
        "de 85,25 %, una sensibilidad de 100 % y un "
        "AUC de 96,96 %."
    )

    st.write(
        "**Red neuronal artificial:** obtuvo una exactitud "
        "de 77,05 %, una sensibilidad de 76,92 % y un "
        "AUC de 82,37 %."
    )

    st.divider()

    st.caption(
        "Los resultados corresponden a una única partición "
        "de entrenamiento y prueba de 61 pacientes. "
        "No representan validación clínica externa."
    )


# =====================================================
# 10. INFORMACION DEL PROYECTO
# =====================================================

elif pagina == "Información del proyecto":

    st.title("Información del proyecto")

    st.write(
        "DENGUE AI es una aplicación web académica "
        "desarrollada para aplicar técnicas de analítica "
        "de datos y aprendizaje automático."
    )

    st.divider()

    st.subheader("Objetivo general")

    st.write(
        "Desarrollar una aplicación web que implemente "
        "modelos de regresión logística y redes neuronales "
        "artificiales para clasificar registros de pacientes "
        "en las categorías de dengue no complicado "
        "y dengue complicado."
    )

    st.divider()

    st.subheader("Base de datos")

    st.write(
        "La base de datos original contiene 303 registros "
        "y 25 variables. Después del proceso de limpieza "
        "se conservaron 302 registros y se generaron "
        "nuevas variables predictoras."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("**Distribución de los registros**")

        st.write("Dengue no complicado: 239 pacientes")

        st.write("Dengue complicado: 63 pacientes")

    with col2:

        st.markdown("**Variables utilizadas**")

        st.write("18 variables predictoras")

        st.write("1 variable objetivo")

    st.divider()

    st.subheader("Modelos utilizados")

    st.markdown("**Regresión logística**")

    st.write(
        "Algoritmo de clasificación supervisada "
        "utilizado para estimar la probabilidad "
        "de pertenencia a una categoría."
    )

    st.markdown("**Red neuronal artificial**")

    st.write(
        "Modelo de aprendizaje automático basado "
        "en una red multicapa con 18 variables de entrada, "
        "dos capas ocultas de 16 y 8 neuronas, "
        "y una salida de clasificación."
    )

    st.divider()

    st.subheader("Tecnologías utilizadas")

    st.write(
        "- Python\n"
        "- Pandas\n"
        "- NumPy\n"
        "- Scikit-learn\n"
        "- Streamlit\n"
        "- Joblib\n"
        "- Matplotlib"
    )

    st.divider()

    st.subheader("Limitaciones")

    st.write(
        "El sistema utiliza una base de datos limitada "
        "y los resultados presentados corresponden "
        "a una evaluación inicial. Se requiere validación "
        "adicional para determinar la capacidad de "
        "generalización de los modelos."
    )

    st.warning(
        "DENGUE AI es exclusivamente un prototipo académico. "
        "No está validado para diagnóstico, triaje "
        "ni decisiones clínicas."
    )


# =====================================================
# 11. PIE DE PAGINA
# =====================================================

st.divider()

st.caption(
    "DENGUE AI | Plataforma académica de analítica de datos"
)

st.caption(
    "Modelos de clasificación: regresión logística "
    "y redes neuronales artificiales."
)
