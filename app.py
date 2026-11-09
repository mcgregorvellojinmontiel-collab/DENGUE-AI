
import streamlit as st
import pandas as pd
import joblib
import io
import base64

from pathlib import Path

# =====================================================
# CONFIGURACIÓN GENERAL
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
ICON_PATH = ASSETS_DIR / "healthicon.png"
BANNER_PATH = ASSETS_DIR / "banner_laboratorio.png"

st.set_page_config(
    page_title="Modelo de predicción de dengue",
    page_icon=str(ICON_PATH),
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
    font-family: "Trebuchet MS", "Segoe UI", sans-serif !important;
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


/* AVISO DE LIMITACIONES DE USO */

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


/* Ocultar navegación lateral y elementos de interfaz de Streamlit */
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
#MainMenu,
footer {
    display: none !important;
    visibility: hidden !important;
}

/* Evitar margen reservado para la barra lateral */
[data-testid="stMainBlockContainer"] {
    padding-top: 1.4rem !important;
}

/* Controles desplegables siempre en tema claro */
div[data-baseweb="select"] > div,
div[data-baseweb="select"] div[role="button"],
[data-baseweb="popover"],
ul[role="listbox"],
ul[role="listbox"] li,
div[role="option"] {
    background-color: #FFFFFF !important;
    color: #263746 !important;
}
div[data-baseweb="select"] svg {
    fill: #263746 !important;
    color: #263746 !important;
}

/* Botones: fondo verde y texto blanco, incluso en elementos internos */
.stButton > button,
.stFormSubmitButton > button,
[data-testid="stFormSubmitButton"] button,
[data-testid="stDownloadButton"] button,
button[kind="primary"],
button[kind="secondary"] {
    background: #1769AA !important;
    background-color: #1769AA !important;
    border: 1px solid #1769AA !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    opacity: 1 !important;
    border-radius: 5px !important;
    font-weight: 600 !important;
}
.stButton > button *,
.stFormSubmitButton > button *,
[data-testid="stFormSubmitButton"] button *,
[data-testid="stDownloadButton"] button *,
button[kind="primary"] *,
button[kind="secondary"] * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    opacity: 1 !important;
}
.stButton > button:hover,
.stFormSubmitButton > button:hover,
[data-testid="stDownloadButton"] button:hover,
button[kind="primary"]:hover {
    background: #0F4F86 !important;
    border-color: #0F4F86 !important;
}


/* Tipografía general de la interfaz */
html, body, .stApp, button, input, textarea, select,
[data-testid="stWidgetLabel"], [data-testid="stMarkdownContainer"] {
    font-family: "Trebuchet MS", "Segoe UI", sans-serif !important;
}


/* Panel de resultados a la derecha */
[data-testid="stHorizontalBlock"] {
    align-items: flex-start !important;
}


/* Ocultar los enlaces automáticos que Streamlit agrega junto a los encabezados */
[data-testid="stHeaderActionElements"],
.stMarkdown [data-testid="stHeaderActionElements"],
[data-testid="stHeadingWithActionElements"] a,
[data-testid="stHeadingWithActionElements"] button {
    display: none !important;
    visibility: hidden !important;
    pointer-events: none !important;
}

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

# =====================================================
# FUNCIONES PARA PREDICCION MASIVA DESDE EXCEL
# =====================================================

COLUMNAS_ORIGINALES = {
    "AGE": "AGE",
    "GENDER": "GENDER",
    "DURATION OF FEVER": "DURATION OF FEVER",
    "HEADACHE": "HEADACHE",
    "MYALGIA": "MYALGIA",
    "ABDOMINAL PAIN": "ABDOMINAL PAIN",
    "RASH": "RASH",
    "VOMITING": "VOMITING",
    "BREATHLESSNESS": "BREATHLESSNESS",
    "BLEEDING": "BLEEDING",
    "BP AT ADMISSION": "BP AT ADMISSION",
    "ORGANOMEGALY": "ORGANOMEGALY",
    "HB": "HB",
    "PCV": "PCV",
    "HIGHEST WBC": "HIGHEST WBC",
    "PLATELET COUNT": "PLATELET COUNT",
    "ALT": "ALT",
    "CREATININE": "CREATININE",
}


def preparar_excel_para_modelos(df_excel):
    """Acepta columnas ya transformadas o el formato de la base original."""
    df = df_excel.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
    )

    # Formato listo para el modelo: las 18 columnas predictoras.
    if all(col in df.columns for col in COLUMNAS):
        X = df[COLUMNAS].copy()
    else:
        # Formato de la base original: convierte las variables 1/2
        # y calcula los indicadores derivados igual que en la predicción individual.
        requeridas = list(COLUMNAS_ORIGINALES.values())
        faltantes = [c for c in requeridas if c not in df.columns]
        if faltantes:
            raise ValueError(
                "El Excel no corresponde al formato esperado. "
                "Faltan columnas: " + ", ".join(faltantes)
            )

        raw = df[requeridas].copy()
        for col in requeridas:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

        # En la base original: 1/2 se codifican como 0/1.
        # AGE: 1 = menor de 6 años; 2 = mayor/igual a 6 años.
        # GENDER: 1 = masculino; 2 = femenino.
        # Síntomas: 1 = presente; 2 = ausente.
        X = pd.DataFrame(index=df.index)
        X["AGE"] = raw["AGE"].map({1: 0, 2: 1})
        X["GENDER"] = raw["GENDER"].map({1: 0, 2: 1})
        X["DURATION OF FEVER"] = raw["DURATION OF FEVER"]

        for col in [
            "HEADACHE", "MYALGIA", "ABDOMINAL PAIN", "RASH",
            "VOMITING", "BREATHLESSNESS", "BLEEDING",
            "BP AT ADMISSION", "ORGANOMEGALY"
        ]:
            X[col] = raw[col].map({1: 1, 2: 0})

        X["HB"] = raw["HB"]
        X["CREATININE"] = raw["CREATININE"]

        # Aplicar la misma imputación usada en la limpieza del dataset.
        # ALT se utiliza para crear ALT_ALTO, por eso se imputa antes
        # de calcular el indicador derivado.
        alt_limpia = raw["ALT"].fillna(26.0)
        X["PCV_ALTO"] = (raw["PCV"] >= 42).astype(int)
        X["PLAQUETAS_BAJAS"] = (raw["PLATELET COUNT"] <= 75000).astype(int)
        X["WBC_ALTO"] = (raw["HIGHEST WBC"] >= 7000).astype(int)
        X["ALT_ALTO"] = (alt_limpia >= 70).astype(int)
        X = X[COLUMNAS]

    X = X.apply(pd.to_numeric, errors="coerce")

    # La limpieza original imputó CREATININE con mediana 0.5.
    # Se aplica también cuando el Excel ya contiene las 18 variables
    # transformadas o cuando procede de la base original.
    X["CREATININE"] = X["CREATININE"].fillna(0.5)

    if X.isna().any().any():
        columnas_error = X.columns[X.isna().any()].tolist()
        raise ValueError(
            "Hay valores vacíos o no numéricos en otras variables: "
            + ", ".join(columnas_error)
            + ". Revisa el Excel antes de volver a cargarlo."
        )

    return X


def ejecutar_prediccion_lote(df_excel, modelos_seleccionados):
    """Devuelve el Excel original con las predicciones agregadas."""
    X = preparar_excel_para_modelos(df_excel)
    resultados = df_excel.copy()

    for nombre, modelo in modelos_seleccionados.items():
        pred = modelo.predict(X).astype(int)
        probabilidades = modelo.predict_proba(X)
        clases = list(modelo.classes_)
        idx_no = clases.index(0)
        idx_si = clases.index(1)

        sufijo = "LOGISTICA" if nombre == "Regresión logística" else "RED_NEURONAL"
        resultados[f"CLASIFICACION_{sufijo}"] = [
            "Dengue complicado" if valor == 1
            else "Dengue no complicado"
            for valor in pred
        ]
        resultados[f"PROB_NO_COMPLICADO_{sufijo}"] = (
            probabilidades[:, idx_no] * 100
        ).round(2)
        resultados[f"PROB_COMPLICADO_{sufijo}"] = (
            probabilidades[:, idx_si] * 100
        ).round(2)

    return resultados


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


# =====================================================
# ENCABEZADO Y CONTENIDO PRINCIPAL
# =====================================================

# Banner a todo lo ancho, conservando la proporción original de la fotografía.
# object-fit: contain evita recortar cualquier parte de la imagen.
if BANNER_PATH.exists():
    banner_b64 = base64.b64encode(BANNER_PATH.read_bytes()).decode("utf-8")
    st.markdown(
        f"""
        <div class="banner-superior">
            <img
                src="data:image/png;base64,{banner_b64}"
                alt="Laboratorio de análisis"
            />
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    """
    <style>

    /* Banner panorámico: menos alto, recortado de forma controlada */
    [data-testid="stMainBlockContainer"] {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    [data-testid="stMainBlockContainer"] {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    .banner-superior {
        width: 100vw;
        margin-left: calc(50% - 50vw);
        margin-top: -3.5rem;
        margin-bottom: 1.45rem;
        padding: 0;
        height: clamp(145px, 16vw, 220px);
        overflow: hidden;
        background: #FFFFFF;
        line-height: 0;
    }
    .banner-superior img {
        display: block;
        width: 100%;
        height: 100%;
        object-fit: cover;
        object-position: center 48%;
        margin: 0;
    }
    @media (max-width: 700px) {
        .banner-superior {
            height: 145px;
            margin-bottom: 1.1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="max-width:1100px; margin:0 auto 26px auto;">
      <div style="color:#1769AA; font-size:12px; font-weight:700;
                  letter-spacing:1.3px; margin-bottom:8px;">
        ANÁLISIS DE DATOS · SALUD
      </div>
      <h1 style="color:#193650; margin:0 0 14px 0;">
        Modelo de predicción de dengue
      </h1>
      <p style="color:#34495E; font-size:16px; line-height:1.7;">
        Esta herramienta permite analizar información clínica y de laboratorio
        relacionada con el dengue y generar estimaciones mediante dos modelos
        de clasificación: regresión logística y red neuronal artificial.
      </p>
      <p style="color:#34495E; font-size:16px; line-height:1.7;">
        El conjunto de datos utilizado contiene 302 registros después del
        proceso de depuración. Para el análisis se emplean 18 variables
        predictoras que incluyen características demográficas, síntomas,
        hallazgos clínicos y resultados de laboratorio.
      </p>
      <p style="color:#34495E; font-size:16px; line-height:1.7;">
        La plataforma permite ingresar los datos de una persona para obtener
        una estimación individual o cargar un archivo Excel para procesar
        varios registros. Los resultados de los modelos pueden consultarse
        y descargarse para facilitar su revisión.
      </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# Área principal: captura a la izquierda y resultados a la derecha.
col_formulario, col_resultados = st.columns(
    [1.35, 0.85],
    gap="large",
    vertical_alignment="top"
)

with col_formulario:
    st.subheader("Predicción individual")
    st.write(
        "Ingrese los datos disponibles del paciente para obtener "
        "las estimaciones de los modelos."
    )

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

with col_resultados:
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

            Estas estimaciones se generan mediante modelos de clasificación. No deben utilizarse para
            diagnosticar, descartar complicaciones ni decidir
            tratamientos médicos.

        </div>
        """, unsafe_allow_html=True)


# =================================================
# PREDICCION MASIVA DESDE EXCEL
# =================================================

st.divider()
st.subheader("Predicción en lote mediante Excel")
st.write(
    "Cargue un archivo Excel con varios pacientes. "
    "Puede utilizar la base original o un archivo que ya contenga "
    "las 18 variables predictoras transformadas."
)

archivo_excel = st.file_uploader(
    "Seleccionar archivo Excel (.xlsx)",
    type=["xlsx"],
    key="carga_excel_lote"
)

modelo_lote = st.selectbox(
    "Modelo para procesar el archivo",
    [
        "Comparar ambos modelos",
        "Regresión logística",
        "Red neuronal artificial"
    ],
    key="selector_modelo_lote"
)

if archivo_excel is not None:
    try:
        vista_excel = pd.read_excel(archivo_excel)
        st.caption(
            f"Archivo cargado correctamente: {len(vista_excel)} registros. "
            "Presione el botón para procesarlos."
        )

        if st.button(
            "Predecir todos los registros",
            type="primary",
            use_container_width=True,
            key="btn_predecir_lote"
        ):
            if modelo_lote == "Regresión logística":
                modelos_lote = {
                    "Regresión logística": modelo_logistico
                }
            elif modelo_lote == "Red neuronal artificial":
                modelos_lote = {
                    "Red neuronal artificial": modelo_neuronal
                }
            else:
                modelos_lote = {
                    "Regresión logística": modelo_logistico,
                    "Red neuronal artificial": modelo_neuronal
                }

            with st.spinner("Procesando los registros del Excel..."):
                resultados_lote = ejecutar_prediccion_lote(
                    vista_excel, modelos_lote
                )

            st.success(
                f"Proceso completado: {len(resultados_lote)} "
                "registros analizados."
            )
            st.dataframe(
                resultados_lote,
                use_container_width=True,
                hide_index=True
            )

            # Resumen de clasificación para cada modelo ejecutado.
            for nombre, modelo in modelos_lote.items():
                sufijo = (
                    "LOGISTICA"
                    if nombre == "Regresión logística"
                    else "RED_NEURONAL"
                )
                columna_clase = f"CLASIFICACION_{sufijo}"
                conteos = resultados_lote[columna_clase].value_counts()

                st.markdown(f"**Resumen: {nombre}**")
                c1, c2 = st.columns(2)
                c1.metric(
                    "Dengue no complicado",
                    int(conteos.get("Dengue no complicado", 0))
                )
                c2.metric(
                    "Dengue complicado",
                    int(conteos.get("Dengue complicado", 0))
                )

            salida = io.BytesIO()
            with pd.ExcelWriter(salida, engine="openpyxl") as writer:
                resultados_lote.to_excel(
                    writer,
                    index=False,
                    sheet_name="Predicciones"
                )

            st.download_button(
                "Descargar resultados en Excel",
                data=salida.getvalue(),
                file_name="resultados_prediccion_lote.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
                use_container_width=True,
                key="descargar_resultado_lote"
            )

    except Exception as error:
        st.error(f"No se pudo procesar el archivo: {error}")
        st.info(
            "Verifique que el archivo tenga las columnas de la base "
            "original o las 18 variables predictoras del modelo."
        )


# =====================================================
